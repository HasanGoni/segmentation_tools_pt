# Fresh-agent trial of `run-segmentation-pipeline` (issue #125)

Question: can an agent with no prior context, given only "here is my data, train and evaluate and predict", reach a trained, evaluated and predicted result by following the repo's skill?

Setup (same for every trial): a new subagent, no conversation context, told to read `AGENTS.md` first and not to edit the repo. Data: 16 synthetic grey 128 px images with 0/255 masks (one bright rectangle each), 3 new 160 px images to predict. CPU only (GPU busy), no internet. The agent logged the skill it loaded, commands, questions, stalls. Date: 2026-11-25. Logs are not committed (scratch); the findings below are the record.

| Trial | Skill version | Result | Stalls (skill gap unless noted) | Fix |
|---|---|---|---|---|
| 1 | before | model, table, masks after 2 failed runs | (1) copying only `train_example.yaml` loses its `data_patch` / `model/` includes; (2) placeholder `model.weights_path` crashes training offline; (3) no word that patch size must fit the image; (4) eval row F1 1.0 / recall 0 not explained | skill sections 1, 2, 4, 5 |
| 2 | after fix 1 | model, table, masks after 1 failed run | (5) `train_example.yaml` has `export.onnx: true`, crash after the last epoch without `onnx`; (6) CPU settings not stated. **Library bug found:** the eval row of trial 1 and 2 was not a model problem: `segpt-eval` read the ground truth of every binary run as empty (PR #129) | skill section 4; #129 |
| 3 | after fix 2 + #129 | model, believable table, masks after 1 failed run | (7) default `batch_size` 16 > 13 train patches: `ZeroDivisionError` in `train_one_epoch`, nothing in the message points at the cause | skill gotcha; clear error is a separate PR |
| 4 | after fix 3 | model, believable table (F1 0.50, recall 1.0, IoU 0.91, 3 val patches), masks after 1 failed run | (8) the section-4 example command omitted `+data.class_names=[defect]`; `segpt-eval` crashed with `TypeError` | example command now has it; first-run model advice added |

What worked every time: `AGENTS.md` section 11 led to the skill; no trial read library source for setup; `segpt-run` (trials 1-3) and the separate commands (trial 4) both reached the end; `segpt-predict` needed no help.

Not met strictly: no trial finished with zero failed runs. Each trial's stall was a different, new gap (8 in total), and each has a skill fix, but the last fix (8) has not been run again by a fresh agent. Two library follow-ups are not part of this slice: a clear error for zero training batches (own PR), and a clear error from `segpt-eval` / `segpt-train` when `data.class_names` is missing.

Caveats: synthetic data (easy rectangles), CPU, 3 val patches, so the evaluation numbers prove the chain, not model quality. `--dry-run` of `segpt-run` cannot catch these stalls; it only prints commands.
