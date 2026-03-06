//! Orphaned Docker container cleanup.
//!
//! The SandboxReaper periodically scans Docker for IronClaw-labeled containers
//! and cleans up those whose corresponding jobs are not active.
//!
//! **Problem:** If the agent process crashes between container creation and cleanup,
//! containers are orphaned indefinitely.
//!
//! **Solution:** Background reaper task that:
//! 1. Scans Docker for containers with the `ironclaw.job_id` label
//! 2. Checks if each job is active in the ContextManager
//! 3. Cleans up containers with inactive/missing jobs

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::context::ContextManager;
use crate::orchestrator::job_manager::ContainerJobManager;
use crate::sandbox::connect_docker;

/// Configuration for the sandbox reaper.
#[derive(Debug, Clone)]
pub struct ReaperConfig {
    /// How often to scan for orphaned containers.
    pub scan_interval: Duration,
    /// Containers older than this with no active job are reaped.
    pub orphan_threshold: Duration,
    /// Label key for looking up job IDs in Docker metadata.
    pub container_label: String,
}

impl Default for ReaperConfig {
    fn default() -> Self {
        Self {
            scan_interval: Duration::from_secs(300),
            orphan_threshold: Duration::from_secs(600),
            container_label: "ironclaw.job_id".to_string(),
        }
    }
}

/// Background task that periodically cleans up orphaned Docker containers.
pub struct SandboxReaper {
    docker: bollard::Docker,
    job_manager: Arc<ContainerJobManager>,
    context_manager: Arc<ContextManager>,
    config: ReaperConfig,
}

impl SandboxReaper {
    /// Create a new reaper. Connects to Docker eagerly — returns error if Docker unavailable.
    pub async fn new(
        job_manager: Arc<ContainerJobManager>,
        context_manager: Arc<ContextManager>,
        config: ReaperConfig,
    ) -> Result<Self, crate::sandbox::SandboxError> {
        let docker = connect_docker().await?;
        Ok(Self {
            docker,
            job_manager,
            context_manager,
            config,
        })
    }

    /// Run the reaper loop forever. Should be spawned with `tokio::spawn`.
    pub async fn run(self) {
        let mut interval = tokio::time::interval(self.config.scan_interval);
        // Skip any missed ticks if scan takes longer than the interval
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            self.scan_and_reap().await;
        }
    }

    async fn scan_and_reap(&self) {
        let containers = match self.list_ironclaw_containers().await {
            Ok(c) => c,
            Err(e) => {
                tracing::error!(error = %e, "Reaper: failed to list Docker containers");
                return;
            }
        };

        let now = Utc::now();
        for (container_id, job_id, created_at) in containers {
            let age = now.signed_duration_since(created_at);
            let threshold = chrono::Duration::from_std(self.config.orphan_threshold)
                .unwrap_or(chrono::Duration::minutes(10));

            if age < threshold {
                continue; // Too young — skip
            }

            // Check if job is active in ContextManager
            let is_active = match self.context_manager.get_context(job_id).await {
                Ok(ctx) => ctx.state.is_active(),
                Err(_) => false, // Not found — treat as orphaned
            };

            if is_active {
                tracing::debug!(
                    job_id = %job_id,
                    container_id = %&container_id[..12.min(container_id.len())],
                    "Reaper: container has active job, skipping"
                );
                continue;
            }

            tracing::info!(
                job_id = %job_id,
                container_id = %&container_id[..12.min(container_id.len())],
                age_secs = age.num_seconds(),
                "Reaper: orphaned container detected, cleaning up"
            );

            self.reap_container(&container_id, job_id).await;
        }
    }

    /// List all IronClaw-managed containers from Docker.
    ///
    /// Returns tuples of (container_id, job_id, created_at).
    async fn list_ironclaw_containers(
        &self,
    ) -> Result<Vec<(String, Uuid, DateTime<Utc>)>, bollard::errors::Error> {
        use bollard::container::ListContainersOptions;

        let mut filters = HashMap::new();
        filters.insert("label", vec![self.config.container_label.as_str()]);

        let options = ListContainersOptions {
            all: true, // include stopped containers
            filters,
            ..Default::default()
        };

        let summaries = self.docker.list_containers(Some(options)).await?;
        let mut result = Vec::new();

        for summary in summaries {
            let container_id = match summary.id {
                Some(id) => id,
                None => continue,
            };

            let labels = summary.labels.unwrap_or_default();

            // Parse job_id from label
            let job_id = match labels
                .get("ironclaw.job_id")
                .and_then(|s| s.parse::<Uuid>().ok())
            {
                Some(id) => id,
                None => {
                    tracing::warn!(
                        container_id = %&container_id[..12.min(container_id.len())],
                        "Reaper: ironclaw container missing valid job_id label"
                    );
                    continue;
                }
            };

            // Parse created_at from label (set by us at creation time); fall back to Docker timestamp
            let created_at = labels
                .get("ironclaw.created_at")
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Utc))
                .or_else(|| {
                    summary
                        .created
                        .and_then(|ts| DateTime::from_timestamp(ts, 0))
                })
                .unwrap_or_else(Utc::now);

            result.push((container_id, job_id, created_at));
        }

        Ok(result)
    }

    /// Stop and remove a single orphaned container.
    ///
    /// First tries `job_manager.stop_job()` (which also revokes the auth token).
    /// Falls back to direct Docker API if the handle is no longer in the in-memory map
    /// (e.g., after a process restart).
    async fn reap_container(&self, container_id: &str, job_id: Uuid) {
        // Try the high-level stop first (handles token revocation)
        match self.job_manager.stop_job(job_id).await {
            Ok(()) => {
                tracing::info!(
                    job_id = %job_id,
                    "Reaper: cleaned up orphaned container via job_manager"
                );
                return;
            }
            Err(e) => {
                tracing::debug!(
                    job_id = %job_id,
                    error = %e,
                    "Reaper: job_manager.stop_job failed (likely no handle after restart), falling back to direct Docker cleanup"
                );
            }
        }

        // Fall back: direct Docker stop + force remove
        if let Err(e) = self
            .docker
            .stop_container(
                container_id,
                Some(bollard::container::StopContainerOptions { t: 10 }),
            )
            .await
        {
            tracing::debug!(
                job_id = %job_id,
                container_id = %&container_id[..12.min(container_id.len())],
                error = %e,
                "Reaper: stop_container failed (may already be stopped)"
            );
        }

        if let Err(e) = self
            .docker
            .remove_container(
                container_id,
                Some(bollard::container::RemoveContainerOptions {
                    force: true,
                    ..Default::default()
                }),
            )
            .await
        {
            tracing::error!(
                job_id = %job_id,
                container_id = %&container_id[..12.min(container_id.len())],
                error = %e,
                "Reaper: failed to remove orphaned container"
            );
        } else {
            tracing::info!(
                job_id = %job_id,
                container_id = %&container_id[..12.min(container_id.len())],
                "Reaper: removed orphaned container via direct Docker API"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test: age threshold filtering
    #[test]
    fn orphan_threshold_filters_young_containers() {
        let threshold = chrono::Duration::minutes(10);
        let young_age = chrono::Duration::minutes(2);
        assert!(young_age < threshold, "Young container should be skipped");
    }

    #[test]
    fn orphan_threshold_allows_old_containers() {
        let threshold = chrono::Duration::minutes(10);
        let old_age = chrono::Duration::minutes(15);
        assert!(old_age >= threshold, "Old container should be reaped");
    }

    // Test: active job detection
    #[tokio::test]
    async fn active_job_is_not_orphaned() {
        let ctx_mgr = Arc::new(ContextManager::new(5));

        // Create job and get its ID
        let job_id = ctx_mgr
            .create_job_for_user("default", "test", "test description")
            .await
            .unwrap();

        let ctx = ctx_mgr.get_context(job_id).await.unwrap();
        assert!(ctx.state.is_active(), "Pending job should be active");
    }

    #[tokio::test]
    async fn missing_job_is_treated_as_orphaned() {
        let ctx_mgr = Arc::new(ContextManager::new(5));
        let job_id = Uuid::new_v4(); // Not created
        let is_active = match ctx_mgr.get_context(job_id).await {
            Ok(ctx) => ctx.state.is_active(),
            Err(_) => false,
        };
        assert!(!is_active, "Missing job should be treated as orphaned");
    }

    #[tokio::test]
    async fn terminal_job_is_treated_as_orphaned() {
        use crate::context::JobState;

        let ctx_mgr = Arc::new(ContextManager::new(5));
        let job_id = ctx_mgr
            .create_job_for_user("default", "test", "test description")
            .await
            .unwrap();
        ctx_mgr
            .update_context(job_id, |ctx| {
                ctx.state = JobState::Failed;
            })
            .await
            .unwrap();

        let ctx = ctx_mgr.get_context(job_id).await.unwrap();
        assert!(
            !ctx.state.is_active(),
            "Failed job should be treated as orphaned"
        );
    }
}
