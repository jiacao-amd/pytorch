#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

BASE_TREE="${BASE_TREE:-}"
HEAD_TREE="${HEAD_TREE:-}"
RUNTIME_TREE="${RUNTIME_TREE:-}"
BUILD_TREE="${BUILD_TREE:-}"
RESULT_ROOT="${RESULT_ROOT:-}"
GPU_IDS="${GPU_IDS:-0}"
IMAGE="${IMAGE:-ato-flydsl-flex:0.3.1}"
CONTAINER_PYTHON="${CONTAINER_PYTHON:-/opt/venv/bin/python}"
CASE_REGEX="${CASE_REGEX:-}"
ROUNDS="${ROUNDS:-2}"
RESUME="${RESUME:-0}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
RUN_PREFILL=1
RUN_DECODE=1
SUMMARIZE_ONLY=0

PREFILL_WARMUP="${PREFILL_WARMUP:-50}"
PREFILL_REPEATS="${PREFILL_REPEATS:-21}"
PREFILL_LAUNCHES="${PREFILL_LAUNCHES:-100}"
DECODE_WARMUP="${DECODE_WARMUP:-100}"
DECODE_REPEATS="${DECODE_REPEATS:-31}"
DECODE_LAUNCHES="${DECODE_LAUNCHES:-200}"
MAX_OUTPUT_ABS="${MAX_OUTPUT_ABS:-0.02}"

BASE_EXPECTED_COMMIT="${BASE_EXPECTED_COMMIT:-a0162f6c53e51f6a96a932dc0a691225c0938553}"
HEAD_EXPECTED_COMMIT="${HEAD_EXPECTED_COMMIT:-420a10ffeb2c19dfd27ffd54bab20cfe982cc54e}"
RUNTIME_EXPECTED_COMMIT="${RUNTIME_EXPECTED_COMMIT:-2a97c4c86f5dc6d528e51946c03634e49af9eb92}"
EXPECTED_TORCH="${EXPECTED_TORCH:-2.15.0a0+git2a97c4c}"
EXPECTED_TORCH_GIT="${EXPECTED_TORCH_GIT:-2a97c4c86f5dc6d528e51946c03634e49af9eb92}"
EXPECTED_ROCM="${EXPECTED_ROCM:-7.2.53211}"
EXPECTED_FLYDSL="${EXPECTED_FLYDSL:-0.3.1}"
EXPECTED_ARCH_PREFIX="${EXPECTED_ARCH_PREFIX:-gfx950}"

usage() {
    cat <<'EOF'
Usage:
  run_ab.sh --base-tree PATH --head-tree PATH --runtime-tree PATH
            --build-tree PATH [options]

Required:
  --base-tree PATH       PyTorch worktree for the parent/base commit
  --head-tree PATH       PyTorch worktree for the last/head commit
  --runtime-tree PATH    PyTorch source tree for the pinned binary runtime
  --build-tree PATH      Matching PyTorch build tree

Options:
  --gpus LIST            Comma-separated physical GPU IDs (default: 0)
  --results PATH         New output directory
  --image IMAGE          Docker image (default: ato-flydsl-flex:0.3.1)
  --rounds N             A/B rounds (default: 2)
  --case-regex REGEX     Run matching case IDs only
  --prefill-only         Skip the decode case
  --decode-only          Skip all prefill cases
  --resume               Skip JSON files already present
  --summarize-only       Rebuild summary from an existing result directory
  -h, --help             Show this help

Timing controls can be overridden with:
  PREFILL_WARMUP, PREFILL_REPEATS, PREFILL_LAUNCHES
  DECODE_WARMUP, DECODE_REPEATS, DECODE_LAUNCHES

Expected commit and software versions can be overridden with:
  BASE_EXPECTED_COMMIT, HEAD_EXPECTED_COMMIT, RUNTIME_EXPECTED_COMMIT
  EXPECTED_TORCH, EXPECTED_TORCH_GIT, EXPECTED_ROCM, EXPECTED_FLYDSL
  EXPECTED_ARCH_PREFIX
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        --base-tree)
            BASE_TREE="$2"
            shift 2
            ;;
        --head-tree)
            HEAD_TREE="$2"
            shift 2
            ;;
        --runtime-tree)
            RUNTIME_TREE="$2"
            shift 2
            ;;
        --build-tree)
            BUILD_TREE="$2"
            shift 2
            ;;
        --gpus)
            GPU_IDS="$2"
            shift 2
            ;;
        --results)
            RESULT_ROOT="$2"
            shift 2
            ;;
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --rounds)
            ROUNDS="$2"
            shift 2
            ;;
        --case-regex)
            CASE_REGEX="$2"
            shift 2
            ;;
        --prefill-only)
            RUN_DECODE=0
            shift
            ;;
        --decode-only)
            RUN_PREFILL=0
            shift
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        --summarize-only)
            SUMMARIZE_ONLY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ "$ROUNDS" =~ ^[1-9][0-9]*$ ]] || die "--rounds must be positive"
[[ "$PREFILL_WARMUP" =~ ^[0-9]+$ ]] || die "invalid PREFILL_WARMUP"
[[ "$PREFILL_REPEATS" =~ ^[1-9][0-9]*$ ]] || die "invalid PREFILL_REPEATS"
[[ "$PREFILL_LAUNCHES" =~ ^[1-9][0-9]*$ ]] || die "invalid PREFILL_LAUNCHES"
[[ "$DECODE_WARMUP" =~ ^[0-9]+$ ]] || die "invalid DECODE_WARMUP"
[[ "$DECODE_REPEATS" =~ ^[1-9][0-9]*$ ]] || die "invalid DECODE_REPEATS"
[[ "$DECODE_LAUNCHES" =~ ^[1-9][0-9]*$ ]] || die "invalid DECODE_LAUNCHES"

if [[ -z "$RESULT_ROOT" ]]; then
    RESULT_ROOT="$PWD/flex-fwd-ab-$(date -u +%Y%m%dT%H%M%SZ)"
fi
RESULT_ROOT="$(realpath -m "$RESULT_ROOT")"

if ((SUMMARIZE_ONLY)); then
    [[ -d "$RESULT_ROOT" ]] || die "result directory does not exist: $RESULT_ROOT"
    exec python3 "$SCRIPT_DIR/summarize.py" \
        --results "$RESULT_ROOT" \
        --rounds "$ROUNDS" \
        --max-output-abs "$MAX_OUTPUT_ABS"
fi

[[ -n "$BASE_TREE" ]] || die "--base-tree is required"
[[ -n "$HEAD_TREE" ]] || die "--head-tree is required"
[[ -n "$RUNTIME_TREE" ]] || die "--runtime-tree is required"
[[ -n "$BUILD_TREE" ]] || die "--build-tree is required"

BASE_TREE="$(realpath -e "$BASE_TREE")"
HEAD_TREE="$(realpath -e "$HEAD_TREE")"
RUNTIME_TREE="$(realpath -e "$RUNTIME_TREE")"
BUILD_TREE="$(realpath -e "$BUILD_TREE")"

command -v docker >/dev/null || die "docker was not found"
command -v git >/dev/null || die "git was not found"
command -v python3 >/dev/null || die "python3 was not found"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "Docker image not found: $IMAGE"
[[ -e /dev/kfd ]] || die "/dev/kfd is not available"
[[ -d /dev/dri ]] || die "/dev/dri is not available"

check_commit() {
    local label="$1"
    local tree="$2"
    local expected="$3"
    local actual
    actual="$(git -C "$tree" rev-parse HEAD)"
    if [[ -n "$expected" && "$actual" != "$expected" ]]; then
        die "$label commit mismatch: expected $expected, found $actual"
    fi
}

check_clean() {
    local label="$1"
    local tree="$2"
    if ((ALLOW_DIRTY)); then
        return
    fi
    if ! git -C "$tree" diff --quiet HEAD -- ||
        ! git -C "$tree" diff --cached --quiet HEAD --; then
        die "$label has tracked changes; set ALLOW_DIRTY=1 only if intentional"
    fi
}

check_commit "base tree" "$BASE_TREE" "$BASE_EXPECTED_COMMIT"
check_commit "head tree" "$HEAD_TREE" "$HEAD_EXPECTED_COMMIT"
check_commit "runtime tree" "$RUNTIME_TREE" "$RUNTIME_EXPECTED_COMMIT"
check_commit "build tree" "$BUILD_TREE" "$RUNTIME_EXPECTED_COMMIT"
check_clean "base tree" "$BASE_TREE"
check_clean "head tree" "$HEAD_TREE"

actual_parent="$(git -C "$HEAD_TREE" rev-parse HEAD^)"
if [[ "$actual_parent" != "$(git -C "$BASE_TREE" rev-parse HEAD)" ]]; then
    die "base tree is not the direct parent of head tree"
fi

mapfile -t torch_c_candidates < <(
    find "$BUILD_TREE/build/torch" -maxdepth 1 -type f -name '_C*.so' | sort
)
((${#torch_c_candidates[@]})) || die "no build/torch/_C*.so found"
TORCH_C_SO="${TORCH_C_SO:-${torch_c_candidates[0]}}"
[[ -f "$TORCH_C_SO" ]] || die "PyTorch extension not found: $TORCH_C_SO"
[[ -f "$BUILD_TREE/torch/version.py" ]] ||
    die "missing $BUILD_TREE/torch/version.py"
[[ -d "$BUILD_TREE/build/lib" ]] || die "missing $BUILD_TREE/build/lib"
[[ -f "$RUNTIME_TREE/torch/__init__.py" ]] ||
    die "invalid runtime tree: $RUNTIME_TREE"

integration_files=(
    torch/_inductor/kernel/flex/flex_attention.py
    torch/_inductor/kernel/flex/flex_flash_attention.py
    torch/_inductor/kernel/flex/flex_flydsl_attention.py
    torch/_inductor/kernel/flex/flex_flydsl_mask.py
    torch/_inductor/kernel/flex/templates/flydsl_forward.py.jinja
    torch/nn/attention/flex_attention.py
)
for tree in "$BASE_TREE" "$HEAD_TREE"; do
    for relative_path in "${integration_files[@]}"; do
        [[ -f "$tree/$relative_path" ]] ||
            die "missing source file: $tree/$relative_path"
    done
    [[ -d "$tree/torch/_inductor/kernel/vendored_templates/flydsl" ]] ||
        die "missing FlyDSL templates in $tree"
done

GPU_IDS="${GPU_IDS//[[:space:]]/}"
IFS=',' read -r -a GPUS <<<"$GPU_IDS"
((${#GPUS[@]})) || die "no GPUs selected"
for gpu in "${GPUS[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU ID: $gpu"
done

if [[ -d "$RESULT_ROOT" && -n "$(find "$RESULT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]] &&
    ((RESUME == 0)); then
    die "result directory is not empty; use a new path or --resume: $RESULT_ROOT"
fi

CACHE_ROOT="${CACHE_ROOT:-$RESULT_ROOT/.inductor-cache}"
mkdir -p "$RESULT_ROOT/logs" "$CACHE_ROOT"

prefill_cases=(
    official_b2_hq16_hkv16_s512_d128_causal
    official_b2_hq16_hkv16_s1024_d128_causal
    official_b2_hq16_hkv16_s4096_d128_causal
    official_b2_hq16_hkv2_s512_d128_causal
    official_b2_hq16_hkv2_s1024_d128_causal
    official_b2_hq16_hkv2_s4096_d128_causal
    official_b8_hq16_hkv16_s512_d128_causal
    official_b8_hq16_hkv16_s1024_d128_causal
    official_b8_hq16_hkv16_s4096_d128_causal
    official_b8_hq16_hkv2_s512_d128_causal
    official_b8_hq16_hkv2_s1024_d128_causal
    official_b8_hq16_hkv2_s4096_d128_causal
    official_b16_hq16_hkv16_s512_d128_causal
    official_b16_hq16_hkv16_s1024_d128_causal
    official_b16_hq16_hkv16_s4096_d128_causal
    official_b16_hq16_hkv2_s512_d128_causal
    official_b16_hq16_hkv2_s1024_d128_causal
    official_b16_hq16_hkv2_s4096_d128_causal
    minimax_b1_h4_s1024_d128_causal
    minimax_b1_h4_s2048_d128_sliding_window
    minimax_b1_h4_s2048_d128_document
    minimax_b2_h32_s2048_d128_causal
    minimax_b2_h32_s4096_d128_causal
    minimax_b2_h32_s8192_d128_causal
    minimax_b2_h32_s16384_d128_causal
    dsv3_16b_b4_h16_s4096
    dsv2_236b_b1_h128_s4096
    dsv3_671b_b4_h128_s4096
    dsv3_16b_b1_h16_s16384
    dsv3_671b_b1_h128_s16384
)
decode_case="minimax_b32_h64_hkv4_sq4_sk8192_d128_decode_recent16"

case_selected() {
    local case_id="$1"
    [[ -z "$CASE_REGEX" || "$case_id" =~ $CASE_REGEX ]]
}

selected_count=0
if ((RUN_PREFILL)); then
    for case_id in "${prefill_cases[@]}"; do
        if case_selected "$case_id"; then
            ((selected_count += 1))
        fi
    done
fi
if ((RUN_DECODE)) && case_selected "$decode_case"; then
    ((selected_count += 1))
fi
((selected_count)) || die "no case matched the requested selection"

base_sha="$(git -C "$BASE_TREE" rev-parse HEAD)"
head_sha="$(git -C "$HEAD_TREE" rev-parse HEAD)"
runtime_sha="$(git -C "$RUNTIME_TREE" rev-parse HEAD)"
image_id="$(docker image inspect "$IMAGE" --format '{{.Id}}')"

python3 - \
    "$RESULT_ROOT/run_metadata.json" \
    "$base_sha" \
    "$head_sha" \
    "$runtime_sha" \
    "$IMAGE" \
    "$image_id" \
    "$GPU_IDS" \
    "$ROUNDS" \
    "$CASE_REGEX" \
    "$PREFILL_WARMUP" \
    "$PREFILL_REPEATS" \
    "$PREFILL_LAUNCHES" \
    "$DECODE_WARMUP" \
    "$DECODE_REPEATS" \
    "$DECODE_LAUNCHES" <<'PY'
import datetime
import json
import pathlib
import socket
import sys

(
    output,
    base,
    head,
    runtime,
    image,
    image_id,
    gpus,
    rounds,
    case_regex,
    prefill_warmup,
    prefill_repeats,
    prefill_launches,
    decode_warmup,
    decode_repeats,
    decode_launches,
) = sys.argv[1:]
metadata = {
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "hostname": socket.gethostname(),
    "commits": {"base": base, "head": head, "runtime": runtime},
    "docker": {"image": image, "image_id": image_id},
    "gpu_ids": gpus.split(","),
    "rounds": int(rounds),
    "case_regex": case_regex or None,
    "prefill": {
        "warmup": int(prefill_warmup),
        "repeats": int(prefill_repeats),
        "launches_per_sample": int(prefill_launches),
    },
    "decode": {
        "warmup": int(decode_warmup),
        "repeats": int(decode_repeats),
        "launches_per_sample": int(decode_launches),
    },
}
pathlib.Path(output).write_text(json.dumps(metadata, indent=2) + "\n")
PY

variant_tree() {
    if [[ "$1" == "base" ]]; then
        printf '%s\n' "$BASE_TREE"
    else
        printf '%s\n' "$HEAD_TREE"
    fi
}

run_container() {
    local gpu="$1"
    local round="$2"
    local variant="$3"
    local case_id="$4"
    local benchmark_script="$5"
    local warmup="$6"
    local repeats="$7"
    local launches="$8"
    local tree
    local result_dir
    local result_file
    local log_file
    local cache_dir
    local container_output
    local relative_path
    local -a docker_args

    tree="$(variant_tree "$variant")"
    result_dir="$RESULT_ROOT/round-$round/$variant"
    result_file="$result_dir/$case_id.json"
    log_file="$RESULT_ROOT/logs/round-${round}_${variant}_${case_id}.log"
    cache_dir="$CACHE_ROOT/round-$round/$variant/$case_id"
    container_output="/results/round-$round/$variant/$case_id.json"

    if ((RESUME)) && [[ -s "$result_file" ]]; then
        printf 'SKIP round=%s gpu=%s variant=%s case=%s\n' \
            "$round" "$gpu" "$variant" "$case_id"
        return
    fi

    mkdir -p "$result_dir" "$cache_dir"
    printf 'START round=%s gpu=%s variant=%s case=%s\n' \
        "$round" "$gpu" "$variant" "$case_id"

    docker_args=(
        run
        --rm
        --entrypoint "$CONTAINER_PYTHON"
        --device=/dev/kfd
        --device=/dev/dri
        --group-add video
        --ipc=host
        -e "HIP_VISIBLE_DEVICES=$gpu"
        -e HSA_NO_SCRATCH_RECLAIM=1
        -e PYTHONDONTWRITEBYTECODE=1
        -e PYTHONHASHSEED=0
        -e OMP_NUM_THREADS=1
        -e PYTHONPATH=/work/pytorch
        -e LD_LIBRARY_PATH=/work/build/lib:/opt/rocm/lib:/opt/rocm/lib64
        -e TORCHINDUCTOR_CACHE_DIR=/cache
        -e "EXPECTED_TORCH=$EXPECTED_TORCH"
        -e "EXPECTED_TORCH_GIT=$EXPECTED_TORCH_GIT"
        -e "EXPECTED_ROCM=$EXPECTED_ROCM"
        -e "EXPECTED_FLYDSL=$EXPECTED_FLYDSL"
        -e "EXPECTED_ARCH_PREFIX=$EXPECTED_ARCH_PREFIX"
        -v "$RUNTIME_TREE:/work/pytorch:ro"
        -v "$TORCH_C_SO:/work/pytorch/torch/$(basename "$TORCH_C_SO"):ro"
        -v "$BUILD_TREE/torch/version.py:/work/pytorch/torch/version.py:ro"
        -v "$BUILD_TREE/build/lib:/work/build/lib:ro"
    )
    for relative_path in "${integration_files[@]}"; do
        docker_args+=(
            -v "$tree/$relative_path:/work/pytorch/$relative_path:ro"
        )
    done
    docker_args+=(
        -v "$tree/torch/_inductor/kernel/vendored_templates/flydsl:/work/pytorch/torch/_inductor/kernel/vendored_templates/flydsl:ro"
        -v "$SCRIPT_DIR:/bench:ro"
        -v "$RESULT_ROOT:/results"
        -v "$cache_dir:/cache"
        -w /tmp
        "$IMAGE"
        "/bench/$benchmark_script"
        --warmup "$warmup"
        --repeats "$repeats"
        --launches "$launches"
        --output "$container_output"
    )
    if [[ "$benchmark_script" == "bench_prefill.py" ]]; then
        docker_args+=(
            --manifest /bench/shapes.json
            --case-id "$case_id"
        )
    fi

    if ! docker "${docker_args[@]}" >"$log_file" 2>&1; then
        printf 'FAILED round=%s gpu=%s variant=%s case=%s log=%s\n' \
            "$round" "$gpu" "$variant" "$case_id" "$log_file" >&2
        tail -n 80 "$log_file" >&2
        return 1
    fi
    [[ -s "$result_file" ]] || {
        printf 'FAILED: result was not written: %s\n' "$result_file" >&2
        return 1
    }
    printf 'DONE round=%s gpu=%s variant=%s case=%s\n' \
        "$round" "$gpu" "$variant" "$case_id"
}

prefill_worker() {
    local gpu="$1"
    local slot="$2"
    local worker_count="$3"
    local round="$4"
    local index
    local case_id
    local -a order

    for index in "${!prefill_cases[@]}"; do
        ((index % worker_count == slot)) || continue
        case_id="${prefill_cases[$index]}"
        case_selected "$case_id" || continue
        if (((index + round) % 2 == 1)); then
            order=(base head)
        else
            order=(head base)
        fi
        for variant in "${order[@]}"; do
            run_container \
                "$gpu" \
                "$round" \
                "$variant" \
                "$case_id" \
                bench_prefill.py \
                "$PREFILL_WARMUP" \
                "$PREFILL_REPEATS" \
                "$PREFILL_LAUNCHES" || return 1
        done
    done
}

run_prefill_round() {
    local round="$1"
    local slot
    local status=0
    local -a pids=()

    for slot in "${!GPUS[@]}"; do
        prefill_worker "${GPUS[$slot]}" "$slot" "${#GPUS[@]}" "$round" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            status=1
        fi
    done
    return "$status"
}

run_decode_round() {
    local round="$1"
    local -a order
    if ((round % 2 == 1)); then
        order=(base head)
    else
        order=(head base)
    fi
    for variant in "${order[@]}"; do
        run_container \
            "${GPUS[0]}" \
            "$round" \
            "$variant" \
            "$decode_case" \
            bench_decode.py \
            "$DECODE_WARMUP" \
            "$DECODE_REPEATS" \
            "$DECODE_LAUNCHES"
    done
}

printf 'Results: %s\n' "$RESULT_ROOT"
printf 'Base: %s\n' "$base_sha"
printf 'Head: %s\n' "$head_sha"
printf 'Runtime: %s\n' "$runtime_sha"
printf 'GPUs: %s\n' "$GPU_IDS"
printf 'Selected cases: %s\n' "$selected_count"

for ((round = 1; round <= ROUNDS; round++)); do
    printf 'ROUND %s/%s\n' "$round" "$ROUNDS"
    if ((RUN_PREFILL)); then
        run_prefill_round "$round" ||
            die "one or more prefill jobs failed in round $round"
    fi
    if ((RUN_DECODE)) && case_selected "$decode_case"; then
        run_decode_round "$round" ||
            die "decode failed in round $round"
    fi
done

python3 "$SCRIPT_DIR/summarize.py" \
    --results "$RESULT_ROOT" \
    --rounds "$ROUNDS" \
    --max-output-abs "$MAX_OUTPUT_ABS"
