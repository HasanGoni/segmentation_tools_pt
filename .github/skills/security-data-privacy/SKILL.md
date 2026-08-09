---
name: security-data-privacy
description: 'PLACEHOLDER skill — data privacy / security practices for industrial segmentation images (anonymization, access control, encryption, audit logging) are NOT implemented in segmentation_tools_pt yet. Use when a user asks "sensitive images", "GDPR", "anonymize data", "encrypt checkpoints", or "access control". Explains considerations and where tooling would be added.'
argument-hint: '[optional] type of sensitive data'
user-invocable: true
disable-model-invocation: false
---
# Security & Data Privacy — NOT IMPLEMENTED YET

Industrial inspection images can contain sensitive product geometry, proprietary designs, or
personally identifiable metadata. Privacy/security tooling is **not implemented** in
`segmentation_tools_pt` today.

## Considerations

| Concern | Mitigation |
|---|---|
| Sensitive imagery in training data | Store on restricted shares; limit read access |
| Metadata leakage | Strip EXIF / file paths from patches and masks |
| Model inversion risk | Restrict checkpoint access; consider differential privacy for public releases |
| Inference logs | Avoid logging full images; log only sample IDs and metrics |
| Encrypted model artifacts | Encrypt `.pth` / `.onnx` at rest if required by policy |

## Where tooling would be added

- New notebook: `nbs/25_security.ipynb` → `security.py`.
- Functions: `strip_exif(image)`, `hash_sample_id(path)`, `encrypt_checkpoint(path, key)`.
- Add config section `security:` with `anonymize_ids`, `encrypt_checkpoints`, `audit_log`.

## What the user can do today

- Keep `raw_dir` and `patch_dir` on access-controlled storage.
- Use UUIDs or hashed IDs for sample names in shared patch caches.
- Review `outputs/` and `checkpoints/` for accidental data leakage before sharing.

## Status

**Not implemented yet.** This skill exists to set expectations and route the user, per the
"placeholder if not in the library" convention.
