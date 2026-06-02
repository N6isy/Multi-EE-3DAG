# MultiEEAffordance annotation package: reviewer_a

This package contains only the samples assigned to `reviewer_a` and the files
referenced by those samples: point clouds, masks, candidate manifests and candidate
npz files.

## How to use locally

1. Extract this archive into the parent directory of your local `MultiEEAffordance/`
   data folder, or merge the extracted `MultiEEAffordance/` directory into your
   project data directory.

2. From the project repository root, start the annotation web app with the reviewer
   sample file below. Adjust the script arguments according to your local version:

```bash
python MultiEEAffordance/tools/serve_v2_annotation_app.py \
  --dataset-root MultiEEAffordance \
  --samples processed/annotation_batches/partnet_adapter_smoke/reviewer_a_samples.jsonl \
  --review-jsonl processed/annotation_batches/partnet_adapter_smoke/reviewer_a_review_records.jsonl \
  --output-mask-root processed/annotation_batches/partnet_adapter_smoke/manual_refined_masks_reviewer_a \
  --output-samples processed/annotation_batches/partnet_adapter_smoke/reviewer_a_refined_samples.jsonl \
  --port 8765 \
  --top-k-candidates 12
```

If your local tool expects an absolute dataset root, use the absolute path to the
extracted `MultiEEAffordance` directory.

## Important

- Do not edit the JSONL file manually.
- Keep the generated annotation outputs under `processed/annotation_batches/`.
- Paths in this package are rewritten to be relative to the local
  `MultiEEAffordance/` directory, so the server path `/home/lzq/data/...` is not
  required on the reviewer's machine.
