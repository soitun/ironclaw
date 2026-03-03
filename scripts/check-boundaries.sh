#!/usr/bin/env bash
# Architecture boundary checks for IronClaw.
# Run as: bash scripts/check-boundaries.sh
# Returns non-zero if hard violations are found.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

violations=0

echo "=== Architecture Boundary Checks ==="
echo

# --------------------------------------------------------------------------
# Check 1: Direct database driver usage outside the db layer
# --------------------------------------------------------------------------
# tokio_postgres:: and libsql:: types should only appear in:
#   - src/db/           (the database abstraction layer)
#   - src/workspace/repository.rs (workspace's own DB layer)
#   - src/error.rs      (needs From impls for driver error types)
#   - src/app.rs        (bootstraps/initialises the database)
#   - src/testing.rs    (test infrastructure)
#   - src/cli/          (CLI commands that bootstrap DB connections)
#   - src/setup/        (onboarding wizard bootstraps DB)
#   - src/main.rs       (entry point)
#
# Everything else is a boundary violation -- those modules should go through
# the Database trait, not touch driver types directly.
# --------------------------------------------------------------------------

echo "--- Check 1: Direct database driver usage outside db layer ---"

results=$(grep -rn 'tokio_postgres::\|libsql::' src/ \
    --include='*.rs' \
    | grep -v 'src/db/' \
    | grep -v 'src/workspace/repository.rs' \
    | grep -v 'src/error.rs' \
    | grep -v 'src/app.rs' \
    | grep -v 'src/testing.rs' \
    | grep -v 'src/cli/' \
    | grep -v 'src/setup/' \
    | grep -v 'src/main.rs' \
    | grep -v '^\s*//' \
    | grep -v '//.*tokio_postgres\|//.*libsql' \
    || true)

if [ -n "$results" ]; then
    echo "VIOLATION: Direct database driver usage found outside db layer:"
    echo "$results"
    echo
    count=$(echo "$results" | wc -l | tr -d ' ')
    echo "($count occurrence(s) -- these modules should use the Database trait)"
    violations=$((violations + 1))
else
    echo "OK"
fi
echo

# --------------------------------------------------------------------------
# Check 2: .unwrap() / .expect() in production code (heuristic)
# --------------------------------------------------------------------------
# We cannot perfectly distinguish test vs production code with grep alone
# (test modules span many lines). Instead we:
#   1. Exclude files that are entirely test infrastructure
#   2. Exclude lines that are clearly in test code (assert, #[test], etc.)
#   3. Report a per-file summary so reviewers can focus on the worst files
#
# This is a WARNING, not a hard violation.
# --------------------------------------------------------------------------

echo "--- Check 2: .unwrap() / .expect() in production code ---"

# Collect raw matches excluding obvious test-only files and lines
raw_results=$(grep -rn '\.unwrap()\|\.expect(' src/ \
    --include='*.rs' \
    | grep -v 'src/main.rs' \
    | grep -v 'src/testing.rs' \
    | grep -v 'src/setup/' \
    || true)

if [ -n "$raw_results" ]; then
    total=$(echo "$raw_results" | wc -l | tr -d ' ')
    echo "WARNING: ~$total .unwrap()/.expect() calls found in src/ (excluding main/testing/setup)."
    echo "Many are in test modules; a per-file breakdown helps triage:"
    echo
    # Show per-file counts, sorted by count descending, top 15
    file_counts=$(echo "$raw_results" | cut -d: -f1 | sort | uniq -c | sort -rn)
    echo "$file_counts" | head -15
    fc_total=$(echo "$file_counts" | wc -l | tr -d ' ')
    if [ "$fc_total" -gt 15 ]; then
        echo "    ... and $((fc_total - 15)) more files"
    fi
    echo
    echo "(This is a warning for gradual cleanup, not a blocking violation.)"
    echo "(Many of these are inside #[cfg(test)] modules which is acceptable.)"
else
    echo "OK"
fi
echo

# --------------------------------------------------------------------------
# Check 3: std::env::var reads outside config/bootstrap layers
# --------------------------------------------------------------------------
# Sensitive values should come through Config or the secrets module.
# Direct std::env::var / env::var() reads are allowed in:
#   - src/config/       (the config layer itself)
#   - src/main.rs       (entry point)
#   - src/setup/        (onboarding wizard)
#   - src/testing.rs    (test infrastructure)
#   - src/cli/          (CLI commands that read env for bootstrap)
#   - src/bootstrap.rs  (bootstrap logic)
# --------------------------------------------------------------------------

echo "--- Check 3: Direct env var reads outside config layer ---"

results=$(grep -rn 'std::env::var\|env::var(' src/ \
    --include='*.rs' \
    | grep -v 'src/config/' \
    | grep -v 'src/main.rs' \
    | grep -v 'src/setup/' \
    | grep -v 'src/testing.rs' \
    | grep -v 'src/cli/' \
    | grep -v 'src/bootstrap.rs' \
    | grep -v '#\[cfg(test)\]' \
    | grep -v '#\[test\]' \
    | grep -v 'mod tests' \
    | grep -v 'fn test_' \
    | grep -v '//.*env::var' \
    || true)

if [ -n "$results" ]; then
    count=$(echo "$results" | wc -l | tr -d ' ')
    echo "WARNING: Direct env var reads found outside config layer ($count occurrences):"
    echo "$results"
    echo
    echo "(Review these -- secrets/config should come through Config or the secrets module)"
else
    echo "OK"
fi
echo

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

echo "=== Summary ==="
if [ "$violations" -gt 0 ]; then
    echo "FAILED: $violations hard violation(s) found"
    exit 1
else
    echo "PASSED: No hard violations found (review warnings above)"
    exit 0
fi
