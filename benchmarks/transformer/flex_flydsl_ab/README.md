# FlyDSL FlexAttention forward A/B benchmark

This bundle reproduces the 31-case forward benchmark used for PR #194309:

- 18 PyTorch-style causal prefill cases
- 7 MiniMax prefill cases
- 5 DSV2/DSV3 TorchTitan prefill cases
- 1 MiniMax sparse decode case

The default A/B pair is:

- base: `a0162f6c53e51f6a96a932dc0a691225c0938553`
- head: `420a10ffeb2c19dfd27ffd54bab20cfe982cc54e`

The runtime is pinned to:

- PyTorch source/build commit:
  `2a97c4c86f5dc6d528e51946c03634e49af9eb92`
- PyTorch version: `2.15.0a0+git2a97c4c`
- ROCm: `7.2.53211`
- FlyDSL: `0.3.1`
- GPU architecture: `gfx950`

## Required directories

Prepare three PyTorch trees:

1. `BASE_TREE`: a clean worktree at the base commit.
2. `HEAD_TREE`: a clean worktree at the head commit.
3. `RUNTIME_TREE`: a clean PyTorch source tree at the pinned runtime commit.

Also provide `BUILD_TREE`, a PyTorch build at the same runtime commit. It must
contain:

- `build/torch/_C*.so`
- `build/lib/`
- `torch/version.py`

The Docker image must contain ROCm and FlyDSL 0.3.1. The default image name is
`ato-flydsl-flex:0.3.1`.

Example worktree setup:

```bash
git -C /path/to/pytorch fetch origin pull/194309/head
git -C /path/to/pytorch worktree add /data/pr194309-head \
  420a10ffeb2c19dfd27ffd54bab20cfe982cc54e
git -C /path/to/pytorch worktree add /data/pr194309-base \
  a0162f6c53e51f6a96a932dc0a691225c0938553
git -C /path/to/pytorch worktree add /data/pytorch-runtime \
  2a97c4c86f5dc6d528e51946c03634e49af9eb92
```

## Run

One GPU:

```bash
./run_ab.sh \
  --base-tree /data/pr194309-base \
  --head-tree /data/pr194309-head \
  --runtime-tree /data/pytorch-runtime \
  --build-tree /data/pytorch-build \
  --gpus 0 \
  --results /data/results/pr194309-ab
```

Eight GPUs:

```bash
./run_ab.sh \
  --base-tree /data/pr194309-base \
  --head-tree /data/pr194309-head \
  --runtime-tree /data/pytorch-runtime \
  --build-tree /data/pytorch-build \
  --gpus 0,1,2,3,4,5,6,7 \
  --results /data/results/pr194309-ab
```

Defaults:

- prefill: 50 warmup iterations, 21 samples, 100 launches per sample
- decode: 100 warmup iterations, 31 samples, 200 launches per sample
- two A/B rounds
- base/head order is reversed in round 2
- every `(round, variant, case)` gets a separate Inductor cache

The script writes:

- raw JSON results under `round-1/` and `round-2/`
- one log per run under `logs/`
- `summary.csv`
- `summary.md`
- `run_metadata.json`

## Useful options

Run only one or a few cases:

```bash
./run_ab.sh ... --case-regex 'official_b2_hq16_hkv2_s512|decode'
```

Run only prefill or decode:

```bash
./run_ab.sh ... --prefill-only
./run_ab.sh ... --decode-only
```

Resume an interrupted result directory:

```bash
./run_ab.sh ... --results /data/results/pr194309-ab --resume
```

Rebuild only the report:

```bash
./run_ab.sh --results /data/results/pr194309-ab --summarize-only
```

To intentionally test different commits or environment versions, override the
expected values:

```bash
BASE_EXPECTED_COMMIT=<sha> \
HEAD_EXPECTED_COMMIT=<sha> \
RUNTIME_EXPECTED_COMMIT=<sha> \
EXPECTED_TORCH=<version> \
EXPECTED_TORCH_GIT=<sha> \
EXPECTED_ROCM=<version> \
EXPECTED_FLYDSL=<version> \
./run_ab.sh ...
```

Do not combine result files produced with different GPUs, software versions,
warmup counts, or launch counts into the same result directory.
