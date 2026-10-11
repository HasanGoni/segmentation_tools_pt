---
name: fast-network-data-copy
description: 'Copy large datasets (potentially millions of images/files) from a network server (e.g. a corporate network share) to a local Windows drive as fast as possible, using a parameterized robocopy wrapper. Use when a user needs to pull image datasets off a slow network share for faster local access, when tuning copy speed/thread counts, when the source/destination path needs to change (all via command-line parameters, no script edits), or when explaining/debugging the copy_images_from_munich.ps1 robocopy wrapper (kept outside this repo).'
argument-hint: '[optional] -Source, -Destination, -FilePatterns, -ThreadCount, -ParallelJobs, -DryRun'
user-invocable: true
disable-model-invocation: false
---

# Fast Network Data Copy

Copy a directory of files (typically images) from a network share to a local drive as fast
as reasonably possible, using `robocopy` under the hood, with every path/pattern/speed knob
exposed as a plain command-line parameter — no script edits needed to point it at a new
source or destination.

Implemented in `copy_images_from_munich.ps1` — **kept outside this repo** (not checked in;
lives with your local scripts). The filename references its original use case (Munich
server image datasets) but the script itself is generic — `-Source` and `-Destination`
accept any UNC path or local path.

## When To Use

- You need to pull a large image (or other file) dataset from a network share to local disk
  for faster training/processing access.
- The source or destination path changed and you just need to point the existing script at
  it — via `-Source`/`-Destination`, never by editing the script.
- You're moving a very large number of files (potentially millions) and need copy speed
  tuned (`-ThreadCount`, `-ParallelJobs`).
- You're debugging why a copy is slow, silently skipping files, or exiting with a failure
  code.

## Before Running — Confirm the Basics

Ask (or infer from context) if not already clear:
1. **Source path** — must be reachable (UNC path or mapped/local path). The script fails
   fast with a clear error if not.
2. **Destination path** — created automatically if missing; confirm it's not accidentally
   pointing somewhere that already holds unrelated data (the script never deletes anything
   locally — no `/MIR` — so this is low-risk, but still worth a sanity check for very large
   copies).
3. **File types** — defaults to common image extensions (`*.png *.jpg *.jpeg *.tif *.tiff
   *.bmp`); override via `-FilePatterns` for other data.
4. **Roughly how much data / how many files** — this determines whether the simple mode
   (`-ParallelJobs 1`, the default) is enough, or whether the parallel-by-subfolder mode is
   worth using (see below).

## How It's Fast

1. **Multithreaded per-process copying (`/MT:<ThreadCount>`)** — robocopy's own internal
   parallelism; the main lever for many small/medium files over a LAN. 16-32 is a good
   starting range; raise it on a fast/stable link, lower it if you see connection errors or
   the source server pushes back.
2. **Unbuffered I/O (`/J`)** — faster for large files on a stable corporate LAN (skips extra
   buffering robocopy would otherwise do). If the link is flaky, `/Z` (restartable mode)
   trades some speed for resilience instead — not currently the default here since it's
   noticeably slower; switch to it only if copies keep failing mid-file.
3. **Fast, bounded retries (`/R:2 /W:5`)** — robocopy's default retry behavior (1,000,000
   retries, 30s apart) turns any transient hiccup into a multi-hour hang. Low retry/wait
   values fail fast instead so you notice and can re-run.
4. **Incremental re-runs (`/XO`)** — skip files that are already up to date locally, so
   re-running the same command after a partial/interrupted copy only transfers what's
   missing or newer. Never `/MIR`: a re-run can't delete anything locally by accident.
5. **Parallel-by-subfolder mode (`-ParallelJobs > 1`)** — for millions of files, a single
   robocopy process (even multithreaded) can be bottlenecked by its own directory-scanning
   and per-connection overhead, especially over SMB. When the source has multiple top-level
   subfolders, running several robocopy *processes* concurrently (one per subfolder,
   throttled to `-ParallelJobs` at a time, plus one extra pass for any loose files directly
   under the source root) works around that ceiling. Each process gets its own log file
   (named after its subfolder) to avoid concurrent writes to one log.
   - **Only helps when the source has multiple top-level subfolders.** A single flat folder
     with millions of files won't benefit — increase `-ThreadCount` instead for that case.
   - Keep `-ThreadCount` lower per job when using `-ParallelJobs > 1` (e.g. 8-16), since
     several processes will each be using that many threads concurrently against the same
     link/server.
6. **Quiet console by default in parallel mode** — per-subfolder jobs don't mirror to the
   console (`/TEE` is reserved for the simple single-job mode) because multiple processes
   writing to one console interleave into a garbled mess; watch the per-job log files or the
   printed per-job exit-code summary instead.

## Changing Source/Destination/Speed — Always Via Parameters

Never hand-edit the script to point it at a new path or dataset. Every knob is a parameter:

```powershell
# Simple mode (default), overriding source/destination
./copy_images_from_munich.ps1 -Source '\\server\share\some\path' -Destination 'D:\local\path'

# Preview only, no copying
./copy_images_from_munich.ps1 -Source '...' -Destination '...' -DryRun

# Different file types
./copy_images_from_munich.ps1 -Source '...' -Destination '...' -FilePatterns '*.raw','*.npy'

# Millions of files across many subfolders: parallel mode
./copy_images_from_munich.ps1 -Source '...' -Destination '...' -ParallelJobs 6 -ThreadCount 12
```

## Validation Checklist

- Always run with `-DryRun` first for an unfamiliar/large source to sanity-check what will
  be copied (robocopy `/L` — lists without transferring).
- Check the final printed exit code: robocopy codes 0-7 are success/informational (see
  https://ss64.com/nt/robocopy-exit.html), 8+ means at least one real failure — the script
  already surfaces this distinction and exits non-zero only for real failures.
- For parallel mode, check every per-subfolder log (or at least the printed per-job exit
  code line) — a single failed subfolder job doesn't stop the others, so a partial failure
  can be easy to miss if you only watch the console.
- Spot-check a file count/size comparison between source and destination for a genuinely
  large transfer rather than trusting the console output alone.

## Common Pitfalls

- **Calling `robocopy` by bare name can fail if `System32` isn't on `PATH`** in some
  shells/launchers — the script calls it by full path (`$env:WINDIR\System32\Robocopy.exe`)
  specifically to avoid this; don't change that back to a bare command name.
- **`-ParallelJobs > 1` against a single flat source folder does nothing useful** — it only
  splits work across top-level *subfolders*; a flat folder is still one job. Raise
  `-ThreadCount` instead for that shape of dataset.
- **Robocopy's default retry settings (`/R:1000000 /W:30`) turn any transient error into an
  effective hang** — always keep `/R`/`/W` set low (already the default here); don't remove
  them when extending the script.
- **`/TEE` on multiple concurrent robocopy processes interleaves output into a garbled
  mess** — this is why parallel-mode sub-jobs don't use it; don't add it back there.
- **This is a personal/host-agnostic utility script, not a place for machine-specific
  shortcuts** — any environment-specific speed tricks (e.g. running the copy from a
  different, faster machine) belong in your own notes, not hardcoded into this script or
  skill.

## Example Prompts

- "Copy the images from `\\server\share\dataset` to `D:\dataset` as fast as possible."
- "This dataset has millions of files across many subfolders, make the copy faster."
- "Change the copy script to grab `.tif` files instead of `.png`." → use `-FilePatterns`,
  don't hand-edit the hardcoded default unless the user wants the default itself changed.
- "Why did my copy fail / seem stuck?" → check the exit code and retry settings first.
