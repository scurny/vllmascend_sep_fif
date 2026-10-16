#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash tools/run_liq_test.sh [options]

Run test_liq.py correctness and buffer-contract checks in an activated vLLM-Ascend/CANN environment.
By default the test uses HBM K/V pools. Use --ddr to exercise registered
memfabric DDR pools; DDR setup failures are reported instead of falling back.

Options:
  --build           Rebuild/install custom ops and the torch extension first.
                    Uses pip --no-deps --no-build-isolation; dependencies must exist.
  --build-op        Build only lightning_indexer_quant (kernel, tiling, ACLNN).
                    Reuse the existing torch extension. Install under LOG_DIR/ops
                    and load that package for this run only. Exclusive with --build.
  --diagnose        With --build-op, compile first-row device score/prefetch logs.
  --device ID       Select one physical NPU via ASCEND_RT_VISIBLE_DEVICES.
                    Otherwise preserve the current device visibility.
  --ddr             Use registered memfabric DDR K/V pools for the prefetch tests.
                    Requires the DDR offload runtime on the target host.
  --python PATH     Python interpreter (default: python).
  --log-dir DIR     Log directory (default: a new directory under /tmp).
  -h, --help        Show this help.

Examples:
  bash tools/run_liq_test.sh --build-op --device 0
  bash tools/run_liq_test.sh --build --device 0
  bash tools/run_liq_test.sh --python /path/to/venv/bin/python
HELP
}

fail() {
    printf '[LIQ] ERROR: %s\n' "$*" >&2
    exit 2
}

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=0
build_op=0
diagnose=0
ddr=0
python_bin=python
device=''
log_dir=''
while (($#)); do
    case "$1" in
        --build) build=1; shift ;;
        --build-op) build_op=1; shift ;;
        --diagnose) diagnose=1; shift ;;
        --ddr) ddr=1; shift ;;
        --device|--python|--log-dir)
            (($# >= 2)) && [[ -n "$2" ]] || fail "$1 requires a value"
            case "$1" in
                --device) device=$2 ;;
                --python) python_bin=$2 ;;
                --log-dir) log_dir=$2 ;;
            esac
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown option: $1 (see --help)" ;;
    esac
done

((build == 0 || build_op == 0)) || fail '--build and --build-op cannot be combined'
((diagnose == 0 || build_op == 1)) || fail '--diagnose requires --build-op'
[[ -z "$device" || "$device" =~ ^[0-9]+$ ]] || fail '--device must be a nonnegative NPU ID'
python_bin=$(command -v "$python_bin") || fail 'Python not found; check --python or activate your environment'
if [[ "$python_bin" != /* ]]; then
    python_bin="$(cd -- "$(dirname -- "$python_bin")" && pwd)/$(basename -- "$python_bin")"
fi
command -v npu-smi >/dev/null || fail 'npu-smi not found; activate the NPU/CANN environment first'
[[ -f "$repo_dir/test_liq.py" ]] || fail "Missing $repo_dir/test_liq.py"

if [[ -z "$log_dir" ]]; then
    log_dir=$(mktemp -d /tmp/liq-test.XXXXXX)
fi
mkdir -p -- "$log_dir"
log_dir=$(cd -- "$log_dir" && pwd)
printf '[LIQ] Logs: %s\n' "$log_dir"
trap 'status=$?; printf "[LIQ] Exit status: %s; logs: %s\n" "$status" "$log_dir"' EXIT

run_logged() {
    local stage=$1
    local -a statuses
    shift
    printf '\n[LIQ] Running %s\n' "$stage"
    set +e
    "$@" 2>&1 | tee "$log_dir/$stage.log"
    statuses=("${PIPESTATUS[@]}")
    set -e
    if ((statuses[0] != 0)); then
        printf '[LIQ] %s failed (exit %s)\n' "$stage" "${statuses[0]}" >&2
        return "${statuses[0]}"
    fi
    return "${statuses[1]}"
}

cd -- "$repo_dir"
if [[ -n "$device" ]]; then
    export ASCEND_RT_VISIBLE_DEVICES="$device"
fi
run_logged npu-info npu-smi info
run_logged preflight "$python_bin" - <<'PY'
import torch
import torch_npu

print("PyTorch:", torch.__version__, "TorchNPU:", torch_npu.__version__)
if not torch.npu.is_available():
    raise SystemExit("NPU unavailable; activate the NPU/CANN environment first")
device_name = torch.npu.get_device_name(0)
print("NPU:", device_name)
if "950" not in device_name:
    raise SystemExit("This test runner requires an Ascend 950 NPU")
PY

vendor_dir="$repo_dir/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
if ((build)); then
    run_logged submodules git submodule update --init --recursive
    run_logged build env COMPILE_CUSTOM_KERNELS=1 VLLM_ASCEND_SKIP_ACLNN_BUILD=0 \
        "$python_bin" -m pip install -e . --no-build-isolation --no-deps -v
elif ((build_op)); then
    [[ ! -e "$log_dir/ops" ]] || fail 'LOG_DIR/ops already exists; choose a fresh --log-dir'
    (
        cd -- "$repo_dir/csrc"
        run_logged build-op bash build.sh --pkg --ops=lightning_indexer_quant \
            --soc=ascend950 --vendor_name=custom -j1 \
            --ops-compile-options "-DLIQ_PREFETCH_DIAGNOSTICS=$diagnose"
    )
    op_installer="$repo_dir/csrc/build/cann-ops-transformer-custom_linux-$(uname -m).run"
    [[ -f "$op_installer" ]] || fail "Single-op installer missing: $op_installer"
    mkdir -p -- "$log_dir/ops"
    run_logged install-op bash "$op_installer" --install-path="$log_dir/ops"
    vendor_dir="$log_dir/ops/vendors/custom_transformer"
fi

[[ -f "$vendor_dir/op_api/lib/libcust_opapi.so" ]] || \
    fail "Custom op library missing: $vendor_dir/op_api/lib/libcust_opapi.so (use --build-op or --build)"
printf '[LIQ] Custom op package: %s\n' "$vendor_dir"
export ASCEND_CUSTOM_OPP_PATH="$vendor_dir${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$vendor_dir/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

run_logged extension "$python_bin" - <<'PY'
from pathlib import Path

from vllm_ascend import vllm_ascend_C

extension = Path(vllm_ascend_C.__file__).resolve()
print("Torch extension:", extension)
if extension.parent != (Path.cwd() / "vllm_ascend").resolve():
    raise SystemExit("Loaded an extension outside this checkout; --build-op requires an existing checkout extension, use --build to install it")
PY

test_command=("$python_bin" -u -X faulthandler "$repo_dir/test_liq.py")
if ((ddr)); then
    printf '[LIQ] Testing registered DDR GVA copies.\n'
    test_command+=(--ddr)
else
    printf '[LIQ] Testing HBM GVA copies.\n'
fi
run_logged test "${test_command[@]}"
printf '[LIQ] Correctness suite completed successfully.\n'
