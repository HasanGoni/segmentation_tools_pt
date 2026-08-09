#!/usr/bin/env bash
# =============================================================================
# LSF/bsub job submitter — reads all settings from the Hydra config file.
#
# Usage:
#   bash hpc/submit.sh configs/train.yaml
#   bash hpc/submit.sh configs/train.yaml train.epochs=80 hpc.queue=gpu_long
#
# Any key=value after the config path overrides the YAML on the fly.
# =============================================================================
set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml> [key=value ...]}"
shift
OVERRIDES=("$@")   # remaining args passed straight to segpt-train

# ---------------------------------------------------------------------------
# Parse hpc.* values from the YAML using Python (already in env).
# Any hpc.key=value CLI override is applied before extraction.
# ---------------------------------------------------------------------------
_yaml_get() {
    # _yaml_get <yaml_file> <dot.path> [override_pairs...]
    local yaml="$1"; local key="$2"; shift 2
    python3 - "$yaml" "$key" "$@" <<'PYEOF'
import sys, re
from pathlib import Path

yaml_file, dotkey = sys.argv[1], sys.argv[2]
overrides = sys.argv[3:]

try:
    import yaml as _yaml
    data = _yaml.safe_load(Path(yaml_file).read_text())
except ImportError:
    # Fallback: minimal key=value YAML parser (no nested lists)
    data = {}
    for line in Path(yaml_file).read_text().splitlines():
        m = re.match(r'^(\s*)(\w+)\s*:\s*(.*)$', line)
        if m:
            data[m.group(2)] = m.group(3).strip().strip('"')

# Apply CLI overrides (key=value format, supports nested hpc.foo=bar)
for ov in overrides:
    if '=' in ov:
        k, v = ov.split('=', 1)
        parts = k.split('.')
        node = data
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v

# Traverse dot path
node = data
for part in dotkey.split('.'):
    if isinstance(node, dict) and part in node:
        node = node[part]
    else:
        node = ''
        break

val = '' if node is None else str(node)
print(val)
PYEOF
}

# Collect hpc.* overrides from CLI (e.g. hpc.queue=gpu_long)
HPC_OVERRIDES=()
for ov in "${OVERRIDES[@]}"; do
    [[ "$ov" == hpc.* ]] && HPC_OVERRIDES+=("$ov")
done

get() { _yaml_get "$CONFIG" "$1" "${HPC_OVERRIDES[@]}"; }

JOB_NAME=$(get hpc.job_name);  [ -z "$JOB_NAME" ] && JOB_NAME="segpt_train"
QUEUE=$(get hpc.queue);        [ -z "$QUEUE" ]     && QUEUE="gpu"
DEVICE=$(get hpc.device);      [ -z "$DEVICE" ]    && DEVICE="gpu"
NUM_GPUS=$(get hpc.num_gpus);  [ -z "$NUM_GPUS" ]  && NUM_GPUS=1
GPU_MEM=$(get hpc.gpu_mem_gb)
GPU_MODE=$(get hpc.gpu_mode);  [ -z "$GPU_MODE" ]  && GPU_MODE="exclusive_process"
NUM_CPUS=$(get hpc.num_cpus);  [ -z "$NUM_CPUS" ]  && NUM_CPUS=8
MEM_GB=$(get hpc.mem_gb);      [ -z "$MEM_GB" ]    && MEM_GB=32
WALLTIME=$(get hpc.walltime);  [ -z "$WALLTIME" ]  && WALLTIME="12:00"
PROJECT=$(get hpc.project)
ENV_ACT=$(get hpc.env_activate)
REPO_DIR=$(get hpc.repo_dir);  [ -z "$REPO_DIR" ]  && REPO_DIR='$HOME/segmentation_tools_pt'

# ---------------------------------------------------------------------------
# Build bsub flags
# ---------------------------------------------------------------------------
BSB=(-J "$JOB_NAME" -q "$QUEUE" -n "$NUM_CPUS" -M "${MEM_GB}GB" -W "$WALLTIME")

if [ "$DEVICE" = "gpu" ]; then
    GPU_SPEC="num=${NUM_GPUS}:mode=${GPU_MODE}"
    [ -n "$GPU_MEM" ] && [ "$GPU_MEM" != "None" ] && [ "$GPU_MEM" != "null" ] \
        && GPU_SPEC="${GPU_SPEC}:gmem=${GPU_MEM}G"
    BSB+=(-gpu "$GPU_SPEC")
fi

[ -n "$PROJECT" ] && BSB+=(-P "$PROJECT")

mkdir -p logs
BSB+=(-o "logs/${JOB_NAME}_%J.out" -e "logs/${JOB_NAME}_%J.err")

# Filter out hpc.* overrides — only training overrides go to segpt-train
TRAIN_OVERRIDES=()
for ov in "${OVERRIDES[@]}"; do
    [[ "$ov" != hpc.* ]] && TRAIN_OVERRIDES+=("$ov")
done

CONFIG_ABS="$(realpath "$CONFIG")"

# ---------------------------------------------------------------------------
# Print summary before submitting
# ---------------------------------------------------------------------------
echo "============================================"
echo "  Submitting: $JOB_NAME"
echo "  Queue      : $QUEUE  |  Device: $DEVICE"
[ "$DEVICE" = "gpu" ] && echo "  GPUs       : $NUM_GPUS  mode=$GPU_MODE  gmem=${GPU_MEM:-any}GB"
echo "  CPUs       : $NUM_CPUS  |  Mem: ${MEM_GB}GB  |  Wall: $WALLTIME"
[ -n "$PROJECT" ] && echo "  Project    : $PROJECT"
echo "  Config     : $CONFIG_ABS"
[ ${#TRAIN_OVERRIDES[@]} -gt 0 ] && echo "  Overrides  : ${TRAIN_OVERRIDES[*]}"
echo "============================================"

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
PATCH_DIR=$(get data.patch_dir)

bsub "${BSB[@]}" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
echo "=== Job \${LSB_JOBID} | \$(hostname) | \$(date) ==="
[ "$DEVICE" = "gpu" ] && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source "\$HOME/.bashrc"
${ENV_ACT}

# Set TMPDIR AFTER .bashrc so it cannot be overridden back to an NFS path.
# Multiprocessing workers write shared-memory temp files here; NFS paths cause
# "Device or resource busy" OSErrors at shutdown.
export TMPDIR=/tmp

cd "${REPO_DIR}"
uv sync --quiet

# Copy patch dataset to local node storage for fast I/O.
# NFS reads are slow; local /tmp is on SSD and ~10× faster per batch.
EXTRA_OVERRIDES=""
if [ -n "${PATCH_DIR}" ] && [ -d "${PATCH_DIR}" ]; then
    LOCAL_PATCH_DIR="\$(mktemp -d /tmp/segpt_patches_XXXXXX)"
    echo "[step] copying patches to local node: \${LOCAL_PATCH_DIR}"
    cp -r "${PATCH_DIR}/." "\${LOCAL_PATCH_DIR}/"
    echo "[ok]   patch copy done"
    EXTRA_OVERRIDES="data.patch_dir=\${LOCAL_PATCH_DIR}"
fi

segpt-train "${CONFIG_ABS}" ${TRAIN_OVERRIDES[*]+"${TRAIN_OVERRIDES[*]}"} \${EXTRA_OVERRIDES}

echo "=== Done at \$(date) ==="
JOBSCRIPT
