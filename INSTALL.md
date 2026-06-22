# Installation

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (`pip install uv` or `curl -LsSf https://astral.sh/uv | sh`)
- CUDA-compatible GPU recommended (CPU works but training will be slow)

---

## Steps

### 1. Clone the repository

```bash
git clone https://github.com/Arnaubiosca15/quantifying_emergence_llms.git
cd quantifying_emergence_llms
```

### 2. Clone Easy-Transformer (required dependency, not on PyPI)

```bash
mkdir -p external
git clone https://github.com/redwoodresearch/Easy-Transformer external/Easy-Transformer
```

### 3. Install dependencies

```bash
uv sync
```

This reads `pyproject.toml`, resolves compatible versions automatically, and creates a virtual environment under `.venv/`.

---

## Running scripts

Always prefix with `uv run` so it uses the managed environment:

```bash
uv run python experiments/nis_experiments/single_head_run.py
```

Or activate the environment manually:

```bash
source .venv/bin/activate
python experiments/nis_experiments/single_head_run.py
```

---

## Adding new dependencies

```bash
uv add <package>
```

This updates `pyproject.toml` and `uv.lock` automatically.
