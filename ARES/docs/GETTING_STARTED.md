# Getting Started

This repository (ARES / “ARES” in some older docs) generates **bit-exact INT8** GAP9 projects from Brevitas-quantized PyTorch models.

## Prerequisites (checklist)

- [ ] **Python**: 3.9+ (recommended: 3.10+)
- [ ] **System tools**: `git`, a C toolchain (for GAP9 builds), and enough disk space for `tests/outputs/`
- [ ] **GAP SDK**: installed + working `gapy`/GVSOC (only required if you want to run GAP9 tests)
- [ ] **Conda or venv**: optional but recommended for isolation

## 5-minute installation

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Verify your setup

### 1) Verify Python deps

```bash
python -c "import torch, brevitas, mako, numpy; print('OK', torch.__version__)"
```

Expected output looks like:

```
OK 2.8.0
```

### 2) (Optional) Verify GAP9 SDK + GVSOC

This repo includes an environment helper at `tools/gap9_env_gvsoc.sh`, but it is **machine-specific** and may require editing for your GAP SDK install.

```bash
bash -lc "source tools/gap9_env_gvsoc.sh && which gapy"
```

Expected output looks like:

```
/path/to/gap_sdk/.../gapy
```

## First test run walkthrough (test_1)

### A) Generate the test (Python-only, no GAP SDK required)

```bash
python tests/generate_all_tests.py --test test_1_simplecnn --skip-gvsoc
```

You should see a folder created at:

- `tests/outputs/test_1_simplecnn/`

### B) Run the generated project on GAP9 GVSOC (requires GAP SDK)

```bash
bash -lc "python tests/run_gap9_projects.py --tests test_1_simplecnn"
```

If you prefer the manual route:

```bash
cd tests/outputs/test_1_simplecnn/generated
bash -lc "source tools/gap9_env_gvsoc.sh && make clean all platform=gvsoc && make run platform=gvsoc"
```

## Check compatibility before full generation

Validate models against extractor support before running the full pipeline:

```bash
python tools/check_model_compatibility.py --model-file my_model.py --model-class MyModel
```

For in-repo test networks:

```bash
python tools/check_model_compatibility.py --test-network test_13_transformer_simple --model-class SimpleTransformer
```

See `docs/SUPPORTED_OPERATIONS.md` for the current support matrix.

## Glossary (quick)

- **L1 / L2 / L3**: GAP9 memory hierarchy (L1 is tiny + fast, L3 is huge + slow).
- **Tiling**: splitting an op into L1-sized chunks (“tiles”) to fit compute + buffers.
- **Slab**: larger L3→L2 streaming chunk (outer loop) used when weights/acts don’t fit in L2.
- **Arena**: a single contiguous L2 allocation where activation buffers get static offsets.
- **FC**: Fabric Controller core (runs `main()`, manages L3/filesystem, launches cluster work).
- **Cluster**: 9-core compute domain (Core 8 orchestrates, Cores 0–7 execute parallel work).
- **GVSOC**: GAP SDK simulator for running GAP9 code.

## What to read next

- `docs/ARCHITECTURE.md` — concepts, pipeline, and where code lives.
- `docs/ADDING_OPERATIONS.md` — end-to-end “add a new op” checklist.
- `docs/SUPPORTED_OPERATIONS.md` — supported operations and migration guidance.
- `docs/README.md` — documentation hub + path mapping.