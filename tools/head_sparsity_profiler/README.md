# Head sparsity profiler

This tool reproduces the ShadowNPU AE profiling procedure:

1. split WikiText validation into complete 128-token windows;
2. ablate each attention head and record perplexity;
3. bypass each decoder layer and record perplexity;
4. rank `min(head_ppl * layer_ppl, 102400)` and allocate the retention budget.

Profile values are retention ratios. `0.2` keeps 20% of positions and means
80% sparsity.

## Profile a checkpoint

Install the dependencies, then run one profile per checkpoint:

```bash
python3 -m pip install -r tools/head_sparsity_profiler/requirements.txt

python3 tools/head_sparsity_profiler/head_sparsity_profiler.py collect \
  --model /path/to/model \
  --model-label MODEL_NAME \
  --calibration-text /path/to/wiki.valid.txt \
  --output-dir results/MODEL_NAME-profile \
  --average-retention 0.2 \
  --local-files-only \
  --resume
```

Defaults match the AE ablation mechanics: seed 42, bfloat16, context 128,
batch 128, chunk 32, all complete validation windows, and importance clamp
102400. A positive `--max-samples` is only for smoke tests. Without
`--activation-scales`, the command profiles the loaded checkpoint directly;
pass the AE activation-distribution JSON to reproduce its static W8A8 model.

The command checkpoints after every ablation. `--resume` safely continues an
interrupted run. The output directory contains:

- `head-retention.txt`: runtime per-head retention profile;
- `measurements.json`: resumable measurements and metadata;
- `heads.txt` and `layers.txt`: AE-compatible raw results;
- `profile-report.json`: achieved retention and sparsity.

Use `--artifact-zip /path/to/ShadowNPU.zip` instead of `--calibration-text` to
read the corpus directly from the AE archive.

## Convert existing measurements

```bash
python3 tools/head_sparsity_profiler/head_sparsity_profiler.py convert \
  --head-results heads.txt --layer-results layers.txt \
  --model-label MODEL_NAME --average-retention 0.2 \
  --output head-retention.txt
```

Pass `--measurements measurements.json` instead of the two text files to
reallocate an existing profile to a different retention budget.

## Verify

```bash
python3 -m unittest discover -s tools/head_sparsity_profiler -p 'test_*.py'
```

The checked-in `qwen2-1.5B-retain-0.2.txt` is the published AE profile used by
the packaged Qwen2.5 runner. The historical filename follows the artifact.
It is byte-identical to the published profile: SHA-256
`0425ffef895df428ff70cd013ac108f194831f9d0d77860f4a9ab44f21d29b32`.
The reference implementation is in the
[ShadowNPU AE artifact](https://doi.org/10.5281/zenodo.19555734).
