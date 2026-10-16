#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
scratch=$(mktemp -d /tmp/liq-runner-check.XXXXXX)
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/repo/tools" "$scratch/repo/csrc" "$scratch/bin"
cp "$repo_dir/tools/run_liq_test.sh" "$scratch/repo/tools/run_liq_test.sh"
printf '\n' > "$scratch/repo/test_liq.py"
full_vendor="$scratch/repo/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
mkdir -p "$full_vendor/op_api/lib"
printf 'full-op\n' > "$full_vendor/op_api/lib/libcust_opapi.so"
printf '%s\n' "$full_vendor" > "$scratch/repo/expected-vendor"
: > "$scratch/repo/expected-device"

cat > "$scratch/bin/npu-smi" <<'MOCK'
#!/usr/bin/env bash
printf 'fixture npu-smi\n'
MOCK
cat > "$scratch/bin/git" <<'MOCK'
#!/usr/bin/env bash
printf 'fixture git %s\n' "$*"
MOCK
cat > "$scratch/bin/python" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == '-m' ]]; then
    [[ "$COMPILE_CUSTOM_KERNELS" == 1 && "$VLLM_ASCEND_SKIP_ACLNN_BUILD" == 0 ]]
    touch pip-called
    printf 'fixture full build\n'
    [[ ! -f build-status ]] || exit "$(cat build-status)"
elif [[ "$1" == '-u' ]]; then
    [[ "$PWD" == */repo ]]
    printf '%s\n' "$@" > test-arguments
    vendor=$(cat expected-vendor)
    [[ "${ASCEND_CUSTOM_OPP_PATH}" == "$vendor"* ]]
    [[ "${LD_LIBRARY_PATH}" == "$vendor/op_api/lib"* ]]
    [[ "${ASCEND_RT_VISIBLE_DEVICES:-}" == "$(cat expected-device)" ]]
    printf 'fixture test stdout\n'
    printf 'fixture test stderr\n' >&2
    [[ ! -f test-status ]] || exit "$(cat test-status)"
else
    printf 'fixture python check\n'
fi
MOCK
cat > "$scratch/repo/csrc/build.sh" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
[[ "$PWD" == */repo/csrc ]]
printf '%s\n' "$@" > build-arguments
[[ ! -f ../op-build-status ]] || exit "$(cat ../op-build-status)"
[[ ! -f ../omit-installer ]] || exit 0
mkdir -p build
cp ../installer-fixture "build/cann-ops-transformer-custom_linux-$(uname -m).run"
MOCK
cat > "$scratch/repo/installer-fixture" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == --install-path=* ]]
[[ ! -f install-status ]] || exit "$(cat install-status)"
install_dir=${1#*=}
mkdir -p "$install_dir/vendors/custom_transformer/op_api/lib"
printf 'single-op\n' > "$install_dir/vendors/custom_transformer/op_api/lib/libcust_opapi.so"
printf 'fixture install\n'
MOCK
chmod +x "$scratch/bin/"*
export PATH="$scratch/bin:$PATH"
runner="$scratch/repo/tools/run_liq_test.sh"

check_status() {
    local wanted=$1
    shift
    local actual=0
    "$@" > "$scratch/output" 2>&1 || actual=$?
    if [[ "$actual" != "$wanted" ]]; then
        cat "$scratch/output"
        printf 'expected exit %s, got %s\n' "$wanted" "$actual" >&2
        exit 1
    fi
}

check_status 0 bash "$runner" --help
check_status 2 bash "$runner" --unknown
check_status 2 bash "$runner" --python
check_status 2 bash "$runner" --device invalid
check_status 2 bash "$runner" --log-dir ''
check_status 2 bash "$runner" --python "$scratch/missing"
check_status 2 bash "$runner" --build --build-op

cd /tmp
check_status 0 env LIQ_STAGE=1 bash "$runner" --log-dir "$scratch/test log"
rg -q 'fixture test stdout' "$scratch/test log/test.log"
rg -q 'fixture test stderr' "$scratch/test log/test.log"
[[ ! -f "$scratch/repo/pip-called" ]]
printf '3\n' > "$scratch/repo/expected-device"
check_status 0 bash "$runner" --device 3 --log-dir "$scratch/device"
! rg -qx -- '--ddr' "$scratch/repo/test-arguments"
check_status 0 bash "$runner" --ddr --log-dir "$scratch/ddr"
rg -qx -- '--ddr' "$scratch/repo/test-arguments"
printf '2\n' > "$scratch/repo/expected-device"
check_status 0 env ASCEND_RT_VISIBLE_DEVICES=2 bash "$runner" --log-dir "$scratch/existing-device"
: > "$scratch/repo/expected-device"

op_log="$scratch/op logs"
printf '%s\n' "$op_log/ops/vendors/custom_transformer" > "$scratch/repo/expected-vendor"
check_status 0 bash "$runner" --build-op --log-dir "$op_log"
[[ ! -f "$scratch/repo/pip-called" ]]
[[ "$(cat "$full_vendor/op_api/lib/libcust_opapi.so")" == full-op ]]
[[ "$(cat "$op_log/ops/vendors/custom_transformer/op_api/lib/libcust_opapi.so")" == single-op ]]
rg -qx -- '--ops=lightning_indexer_quant' "$scratch/repo/csrc/build-arguments"
rg -qx -- '--soc=ascend950' "$scratch/repo/csrc/build-arguments"
rg -qx -- '--pkg' "$scratch/repo/csrc/build-arguments"
[[ -f "$op_log/build-op.log" && -f "$op_log/install-op.log" && -f "$op_log/test.log" ]]
check_status 2 bash "$runner" --build-op --log-dir "$op_log"
[[ "$(cat "$op_log/ops/vendors/custom_transformer/op_api/lib/libcust_opapi.so")" == single-op ]]

mv "$full_vendor" "$scratch/full-vendor-hidden"
printf '%s\n' "$scratch/only-op/ops/vendors/custom_transformer" > "$scratch/repo/expected-vendor"
check_status 0 bash "$runner" --build-op --log-dir "$scratch/only-op"
mv "$scratch/full-vendor-hidden" "$full_vendor"

printf '23\n' > "$scratch/repo/op-build-status"
check_status 23 bash "$runner" --build-op --log-dir "$scratch/op-fail"
[[ ! -f "$scratch/op-fail/install-op.log" && ! -f "$scratch/op-fail/test.log" ]]
rm "$scratch/repo/op-build-status"
printf '24\n' > "$scratch/repo/install-status"
check_status 24 bash "$runner" --build-op --log-dir "$scratch/install-fail"
[[ ! -f "$scratch/install-fail/test.log" ]]
rm "$scratch/repo/install-status"
rm "$scratch/repo/csrc/build/cann-ops-transformer-custom_linux-$(uname -m).run"
touch "$scratch/repo/omit-installer"
check_status 2 bash "$runner" --build-op --log-dir "$scratch/missing-package"
[[ ! -f "$scratch/missing-package/test.log" ]]
rm "$scratch/repo/omit-installer"

printf '%s\n' "$full_vendor" > "$scratch/repo/expected-vendor"
check_status 0 bash "$runner" --build --log-dir "$scratch/full-build"
rg -q 'fixture full build' "$scratch/full-build/build.log"
printf '17\n' > "$scratch/repo/build-status"
check_status 17 bash "$runner" --build --log-dir "$scratch/full-build-fail"
[[ ! -f "$scratch/full-build-fail/test.log" ]]
printf '139\n' > "$scratch/repo/test-status"
check_status 139 bash "$runner" --log-dir "$scratch/test-fail"
rg -q 'fixture test stderr' "$scratch/test-fail/test.log"
rm "$scratch/repo/test-status"
printf '139\n' > "$scratch/repo/test-status"
check_status 139 bash "$runner" --ddr --log-dir "$scratch/ddr-test-fail"
rg -q 'fixture test stderr' "$scratch/ddr-test-fail/test.log"
rm "$scratch/repo/test-status"
mv "$full_vendor" "$scratch/vendor-hidden"
check_status 2 bash "$runner" --log-dir "$scratch/no-vendor"
[[ ! -f "$scratch/no-vendor/test.log" ]]
printf 'PASS: single-op isolation, selected build, failure propagation and existing runner modes\n'
