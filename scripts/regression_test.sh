#!/usr/bin/env bash
# CaP-X Regression Test Suite
#
# Usage:
#   ./scripts/regression_test.sh              # Run both tests
#   ./scripts/regression_test.sh test1        # Single-turn only (~3 min)
#   ./scripts/regression_test.sh test2        # Multi-turn VDM only (~15 min)
#   ./scripts/regression_test.sh quick        # 10-trial smoke test (~30 sec)
#
# Prerequisites:
#   API servers must be running on ports 8114 (SAM3), 8115 (GraspNet),
#   8116 (PyRoKi), and 8110/8111 (LLM proxy).
#
# Exit codes:
#   0 = all tests passed
#   1 = one or more tests failed

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TEST1_CONFIG="env_configs/cube_stack/franka_robosuite_cube_stack.yaml"
TEST2_CONFIG="env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm.yaml"

TEST1_MIN_COMPLETED=38   # baseline ~42, allow variance
TEST2_MIN_COMPLETED=70   # baseline ~79, allow variance
QUICK_TRIALS=10
QUICK_MIN_COMPLETED=2    # expect ~4-5 out of 10

WORKERS_TEST1=12
WORKERS_TEST2=8

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
run_test() {
    local name="$1"
    local config="$2"
    local workers="$3"
    local trials="$4"
    local min_completed="$5"

    echo -e "\n${BOLD}━━━ ${name} ━━━${NC}"
    echo "  Config:  ${config}"
    echo "  Trials:  ${trials}  |  Workers: ${workers}  |  Min pass: ${min_completed}"
    echo ""

    local tmpfile
    tmpfile=$(mktemp)

    uv run --no-sync --active capx/envs/launch.py \
        --config-path "$config" \
        --total-trials "$trials" \
        --num-workers "$workers" 2>&1 | tee "$tmpfile"

    # Parse the summary line: "1.000/0.491/49"
    local summary
    summary=$(grep -oP '\d+\.\d+/\d+\.\d+/\d+' "$tmpfile" | tail -1)
    rm -f "$tmpfile"

    if [[ -z "$summary" ]]; then
        echo -e "${RED}FAIL${NC} — could not parse summary output"
        return 1
    fi

    local success_rate avg_reward completed
    IFS='/' read -r success_rate avg_reward completed <<< "$summary"

    echo ""
    echo -e "  Result: ${BOLD}${summary}${NC}  (success_rate/avg_reward/completed)"

    if (( completed >= min_completed )); then
        echo -e "  ${GREEN}PASS${NC}  (${completed} >= ${min_completed})"
        return 0
    else
        echo -e "  ${RED}FAIL${NC}  (${completed} < ${min_completed})"
        return 1
    fi
}

check_server() {
    local port="$1"
    local name="$2"
    # Try /health first, fall back to TCP connection check
    if curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1 \
       || curl -so /dev/null -w '' "http://127.0.0.1:${port}/" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} ${name} (port ${port})"
        return 0
    else
        echo -e "  ${RED}✗${NC} ${name} (port ${port})"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo -e "${BOLD}CaP-X Regression Test Suite${NC}"
echo "Git: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown') ($(git diff --quiet 2>/dev/null && echo 'clean' || echo 'dirty'))"
echo ""

# ---------------------------------------------------------------------------
# Auto-launch missing servers
# ---------------------------------------------------------------------------
PIDS_TO_CLEAN=()

start_if_missing() {
    local port="$1"
    local name="$2"
    shift 2
    local cmd=("$@")

    if python3 -c "import socket; s=socket.create_connection(('127.0.0.1', $port), timeout=1); s.close()" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} ${name} (port ${port}) — already running"
        return 0
    fi

    echo -e "  ${YELLOW}→${NC} Starting ${name} (port ${port})..."
    "${cmd[@]}" &
    PIDS_TO_CLEAN+=($!)
    return 0
}

echo "Checking and auto-launching API servers..."
start_if_missing 8114 "SAM3" \
    python -m capx.serving.launch_sam3_server --port 8114 --device cuda --host 127.0.0.1
start_if_missing 8115 "GraspNet" \
    python -m capx.serving.launch_contact_graspnet_server --port 8115 --device cuda --host 127.0.0.1
start_if_missing 8116 "PyRoKi" \
    python -m capx.serving.launch_pyroki_server --port 8116 --host 127.0.0.1

# Check for at least one LLM proxy
llm_ok=false
if curl -sf "http://127.0.0.1:8110/health" > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} LLM proxy — NVIDIA (port 8110)"
    llm_ok=true
fi
if curl -sf "http://127.0.0.1:8111/health" > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} LLM proxy — OpenRouter (port 8111)"
    llm_ok=true
fi
if ! $llm_ok; then
    echo -e "  ${RED}✗${NC} No LLM proxy found (need port 8110 or 8111)"
    echo -e "  Start one with: python capx/serving/openrouter_server.py --key-file .openrouterkey --port 8110"
    exit 1
fi

# Wait for all servers to be ready (up to 120s)
echo ""
echo "Waiting for servers to be ready..."
deadline=$((SECONDS + 120))
all_ready=false
while (( SECONDS < deadline )); do
    ok=true
    for port in 8114 8115 8116; do
        python3 -c "import socket; s=socket.create_connection(('127.0.0.1', $port), timeout=1); s.close()" 2>/dev/null || ok=false
    done
    if $ok; then
        all_ready=true
        break
    fi
    sleep 5
done

if ! $all_ready; then
    echo -e "${RED}ERROR:${NC} Servers did not become ready within 120s"
    for pid in "${PIDS_TO_CLEAN[@]}"; do kill "$pid" 2>/dev/null; done
    exit 1
fi

echo "Checking all servers..."
check_server 8114 "SAM3"       || { echo -e "${RED}FAIL${NC}"; exit 1; }
check_server 8115 "GraspNet"   || { echo -e "${RED}FAIL${NC}"; exit 1; }
check_server 8116 "PyRoKi"     || { echo -e "${RED}FAIL${NC}"; exit 1; }

# Cleanup servers on exit
cleanup_servers() {
    for pid in "${PIDS_TO_CLEAN[@]}"; do
        kill "$pid" 2>/dev/null
    done
}
trap cleanup_servers EXIT

# Determine which tests to run
mode="${1:-both}"
failures=0

case "$mode" in
    test1)
        run_test "Test 1: Single-turn cube stack" "$TEST1_CONFIG" "$WORKERS_TEST1" 100 "$TEST1_MIN_COMPLETED" || ((failures++))
        ;;
    test2)
        run_test "Test 2: Multi-turn VDM" "$TEST2_CONFIG" "$WORKERS_TEST2" 100 "$TEST2_MIN_COMPLETED" || ((failures++))
        ;;
    quick)
        run_test "Quick smoke test" "$TEST1_CONFIG" 5 "$QUICK_TRIALS" "$QUICK_MIN_COMPLETED" || ((failures++))
        ;;
    both)
        run_test "Test 1: Single-turn cube stack" "$TEST1_CONFIG" "$WORKERS_TEST1" 100 "$TEST1_MIN_COMPLETED" || ((failures++))
        run_test "Test 2: Multi-turn VDM" "$TEST2_CONFIG" "$WORKERS_TEST2" 100 "$TEST2_MIN_COMPLETED" || ((failures++))
        ;;
    *)
        echo "Usage: $0 [test1|test2|quick|both]"
        exit 1
        ;;
esac

# Summary
echo ""
echo -e "${BOLD}━━━ Summary ━━━${NC}"
if (( failures == 0 )); then
    echo -e "${GREEN}All tests passed.${NC}"
    exit 0
else
    echo -e "${RED}${failures} test(s) failed.${NC}"
    exit 1
fi
