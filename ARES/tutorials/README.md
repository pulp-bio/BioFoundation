# ARES Tutorials

Interactive tutorials for learning the ARES deployment pipeline.

## Getting Started

| Tutorial | Description | Time |
|----------|-------------|------|
| [getting_started.ipynb](getting_started.ipynb) | End-to-end pipeline: PyTorch → GAP9 | ~30 min |

## Prerequisites

1. **Python environment** with PyTorch and Brevitas:
   ```bash
   conda activate TimeFM  # or your environment
   ```

2. **GAP9 SDK** (optional, for GVSOC simulation):
   ```bash
   source ~/gap_sdk/configs/gap9_v2.sh
   ```

## Running the Tutorials

```bash
cd tutorials/
jupyter notebook getting_started.ipynb
```

Or use JupyterLab:
```bash
jupyter lab getting_started.ipynb
```

## Output Directory

Generated files are saved to `tutorials/outputs/` (gitignored).
