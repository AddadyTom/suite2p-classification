#!/bin/bash
# Standalone Zero-Crash Cache Recomputation Script
# Can be run directly in your native terminal or from Antigravity.

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE_DIR"

if [ -f "$WORKSPACE_DIR/.venv/bin/python" ]; then
    PYTHON_EXEC="$WORKSPACE_DIR/.venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

export PYTHONPATH="$WORKSPACE_DIR:$PYTHONPATH"

echo "================================================================="
echo "   Suite2p Feature Cache Engine (Zero-Crash Chunked Multi-Core)  "
echo "================================================================="

if [ "$#" -eq 0 ]; then
    echo "Usage Examples:"
    echo "  ./recompute.sh --all                     # Recompute all 44 features on all sessions"
    echo "  ./recompute.sh --feature q95             # Recompute only q95 feature (< 5s)"
    echo "  ./recompute.sh --all --workers 4         # Use 4 parallel workers"
    echo ""
    echo "Running with default: --all --workers 4"
    "$PYTHON_EXEC" recompute_cache.py --all --workers 4
else
    "$PYTHON_EXEC" recompute_cache.py "$@"
fi
