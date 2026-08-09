# HPC Job Submission (`segpt-submit`)

All bsub settings live in the `hpc:` section of your `train.yaml`.
The `segpt-submit` command reads them and submits — no editing scripts.

## Install

```bash
uv sync
segpt-submit --help
```

## Two example GPU queues

### `gpu` — multi-GPU shared nodes, filter by GPU memory

```bash
# Submits:
# bsub -Is -q gpu -gpu "num=3:j_exclusive=yes:gmem=24G"
#      -R 'select[ui==<your_resource_group> && osrel>=70 && type==X64LIN] rusage[mem=200G]'
#      -n 8 -o logs/segpt_train_%J.out -e logs/segpt_train_%J.err

segpt-submit configs/train.yaml
```

Config for this queue:
```yaml
hpc:
  queue: gpu
  device: gpu
  num_gpus: 3
  j_exclusive: true
  gpu_mem_gb: 24     # minimum GPU memory per GPU
  host: ""           # leave empty — shared pool
  num_cpus: 8
  mem_gb: 200
```

### `gpu_compute` — big dedicated nodes (40G+ GPUs), pin to host

```bash
# Submits:
# bsub -Is -q gpu_compute -gpu "num=1:j_exclusive=yes"
#      -R 'select[ui==<your_resource_group> && osrel>=70 && type==X64LIN] rusage[mem=200G]'
#      -m gpu-node-01
#      -n 8 -o logs/segpt_train_%J.out -e logs/segpt_train_%J.err

segpt-submit configs/train.yaml hpc.queue=gpu_compute hpc.num_gpus=1 hpc.gpu_mem_gb="" hpc.host=gpu-node-01
```

Config for this queue:
```yaml
hpc:
  queue: gpu_compute
  device: gpu
  num_gpus: 1
  j_exclusive: true
  gpu_mem_gb: ""     # no gmem filter — the node itself has 40G+
  host: gpu-node-01   # or gpu-node-02
  num_cpus: 8
  mem_gb: 200
```

### CPU-only job (patch build, smoke test)

```bash
# Submits:
# bsub -Is -q normal
#      -R 'select[ui==<your_resource_group> && osrel>=70 && type==X64LIN] rusage[mem=200G]'
#      -n 16 -o logs/...

segpt-submit configs/train.yaml hpc.device=cpu hpc.queue=normal hpc.num_cpus=16 train.device=cpu
```

## Full hpc: config reference

```yaml
hpc:
  job_name: segpt_train       # LSF job name (-J)
  queue: gpu                  # gpu | gpu_compute | normal
  device: gpu                 # gpu | cpu
  interactive: true           # true = -Is (stream output to terminal)
                              # false = fire-and-forget (output to log only)
  # GPU (ignored when device=cpu):
  num_gpus: 3                 # number of GPUs
  j_exclusive: true           # true = j_exclusive=yes (no GPU sharing)
  gpu_mem_gb: 24              # min GPU memory GB; "" = no constraint
  host: ""                    # pin to specific node (e.g. gpu-node-01)
  # CPU / memory:
  num_cpus: 8                 # CPU slots (-n)
  mem_gb: 200                 # total RAM → rusage[mem=200G]
  # Site-specific resource select string (rarely changes):
  resource_select: "ui==<your_resource_group> && osrel>=70 && type==X64LIN"
  project: ""                 # -P code; leave empty if not required
  env_activate: "conda activate segpt"
  repo_dir: "$HOME/segmentation_tools_pt"
```

## CLI override examples

```bash
# Use big GPU node
segpt-submit configs/train.yaml hpc.queue=gpu_compute hpc.host=gpu-node-02 hpc.gpu_mem_gb=""

# More GPUs with higher memory requirement
segpt-submit configs/train.yaml hpc.num_gpus=4 hpc.gpu_mem_gb=40

# Train longer
segpt-submit configs/train.yaml train.epochs=100 metrics.best_metric=recall

# Preview without submitting
segpt-submit configs/train.yaml --dry-run
```

## Monitor

```bash
bjobs                   # list your active jobs
bpeek <JOBID>           # tail live stdout
bjobs -l <JOBID>        # full job details
bkill <JOBID>           # cancel
```

Logs: `logs/<job_name>_<JOBID>.out` / `.err`

---

## Troubleshooting

### Issue 1 — `OSError: [Errno 16] Device or resource busy: /home/.../.tmp/pymp-*`

**What happened**

DataLoader worker processes create shared-memory temp directories under `TMPDIR`.
On this HPC, `.bashrc` sets `TMPDIR` to the user's NFS home directory
(`/path/to/jupyter-home/.tmp/`).
NFS filesystems do not support POSIX shared memory (`shm_open`/`mmap`).
When workers exit they cannot delete their own temp dirs, producing noisy
`OSError` tracebacks in the `.err` log.
Training itself may complete successfully, but the noise triggers the HPC
job accounting watchdog on some queue configurations.

**Initial wrong fix (do not use)**

Setting `num_workers: 0` in the config avoids workers entirely — no workers,
no shared memory, no OSError. But it also means the GPU sits idle while the
main process reads PNGs from NFS, runs augmentations, and stacks tensors.
This idle gap exceeds the HPC 5 % GPU-utilisation watchdog (see Issue 2).

**Correct fix (already applied)**

1. `hpc/submit.sh` — `export TMPDIR=/tmp` is set **after** `source "$HOME/.bashrc"`
   so `.bashrc` cannot override it. `/tmp` is local SSD on every compute node.

2. `scripts/train.py` — `os.environ['TMPDIR'] = '/tmp'` at the very first line,
   before any import. This is belt-and-suspenders: even if the shell environment
   leaks through, Python overrides it before any worker is spawned.

**Config setting**
```yaml
train:
  num_workers: 4   # safe once TMPDIR=/tmp is enforced; keep ≥ 1 for GPU utilisation
```

---

### Issue 2 — Job killed by GPU utilisation watchdog (`< 5 % for 5 minutes`)

**What happened**

The HPC enforces a policy: if GPU utilisation stays below 5 % for 5 consecutive
minutes, the job is killed. With `num_workers: 0` (single-process DataLoader)
the GPU is completely idle while the CPU reads PNG patches from NFS, runs
Albumentations, and stacks tensors. NFS latency is high enough that a batch of
16 × 256×256 patches from network storage takes several seconds per batch —
easily keeping the GPU at 0 % for minutes at a time.

**Fix (already applied)**

Three complementary changes in `hpc/submit.sh`:

1. **TMPDIR fix** (see Issue 1) — enables `num_workers ≥ 1` so workers prefetch
   while the GPU is training.

2. **`num_workers: 4`** — 4 background workers overlap data loading with GPU
   computation; the GPU almost never waits for data.

3. **Copy patches to local node storage** — the job script reads `data.patch_dir`
   from the config, copies it to a temp dir on `/tmp` (local SSD) before training,
   and passes `data.patch_dir=<local_path>` as a Hydra override.
   Local SSD reads are ~10× faster than NFS, fully saturating the workers.

You will see these lines at job start in the `.out` log:
```
[step] copying patches to local node: /tmp/segpt_patches_aB3xYz
[ok]   patch copy done
```

---

### Large patch datasets (hundreds of thousands to millions of patches)

If your patch directory is very large (> ~20 GB), the copy step at job start
may take several minutes or longer. Here are the options in order of preference:

**Option A — Pre-stage once, reuse across runs (recommended)**

Copy the patch directory to a shared scratch filesystem that is local or
close to the compute nodes (e.g. `/scratch`, `/work`, or equivalent on your
cluster). Set `data.patch_dir` in `train.yaml` to that path.
The job script detects the path is already on local storage (`/tmp` or
`/scratch`) and skips the copy.

```yaml
data:
  patch_dir: /scratch/goni/segmentation_patches   # fast filesystem
```

You only copy once. All subsequent jobs reuse it without the startup delay.

**Option B — Increase job walltime to absorb copy time**

If you must copy at job start (e.g. scratch is not available), increase
`hpc.walltime` to account for the copy:

```bash
segpt-submit configs/train.yaml hpc.walltime=14:00   # 2 h extra for large copy
```

**Option C — Disable the copy, keep `num_workers` high**

If the patch directory is too large to copy and NFS reads are still fast enough
to feed the GPU, disable the copy by pointing `patch_dir` at the NFS path and
relying on `num_workers: 4` to hide the latency:

```yaml
data:
  patch_dir: /path/to/dataset/patches   # NFS — no copy
train:
  num_workers: 6   # more workers to compensate for NFS latency
```

Monitor GPU utilisation with `bpeek <JOBID>` or `watch -n5 nvidia-smi` in an
interactive session to confirm it stays above 40 %.

**Option D — Use direct mode (`data.patch_mode: false`)**

For very large datasets where patch extraction itself is the bottleneck,
skip the patch cache entirely. The dataset loads full-size images and applies
a random crop at train time. No patch build step, no large cache to copy.

```yaml
data:
  patch_mode: false     # load full images, crop on-the-fly
  raw_dir: /path/to/raw
```

Trade-off: no hard-example oversampling (weighted sampler requires patch metadata).
