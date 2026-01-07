# MPP Bench (Submission Version)

> **Note:** This submission version contains only a subset of the full dataset due to file size constraints. The complete benchmark will be released separately.

This folder is a self-contained copy of the pieces from `mpp_bench` that are
required to reproduce the evaluation and mitigation pipeline that was used for
our MPP Bench submission.  It bundles the cleaned data, prompts, Python source,
and helper scripts so that a `git clone` followed by dependency installation is
all that is needed to rerun the experiments locally.

## Repository Layout

- `data.zip`, `data.z01` ~ `data.z06` – split compressed archives containing the final benchmark JSON plus the supporting PDFs in `docs_clip/`. See [Data Setup](#data-setup) for extraction instructions.
- `prompts/` – system/user prompt templates referenced by the Python code.
- `src/` – async OpenRouter evaluation, judging, and mitigation drivers.
- `scripts/` – bash wrappers for the submission pipeline (no Slurm or secrets).
- `results/`, `judge_results/`, `metrics/`, `logs/` – empty folders that scripts
  will populate as you run evaluations.

## Data Setup

The data is split into multiple compressed archives (`data.zip`, `data.z01` ~ `data.z06`) due to GitHub file size limits. Extract them before running the pipeline:

```bash
zip -s 0 data.zip --out data_combined.zip
unzip data_combined.zip
rm data_combined.zip  # optional: remove the combined zip after extraction
```

This will create the `data/` directory with the following structure:
```
data/
├── 02_final_faithfulness_checklists.json   # benchmark JSON
└── docs_clip/                               # PDF documents used in evaluation
    ├── *.pdf
    └── ...
```

The evaluation scripts expect PDFs to be located at `data/docs_clip/`.

## Environment Setup

1. Use Python 3.10+.
2. Create a virtual environment (optional but recommended) and install deps:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Export your OpenRouter credentials (no keys are stored in this repo):

   ```bash
   export OPENROUTER_API_KEY="sk-or-your-key"
   # Optional override, defaults to https://openrouter.ai/api/v1
   export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
   ```

You can also set `PYTHON_BIN` if you want the bash helpers to call something
other than `python` (e.g., `PYTHON_BIN=python3`).

## Running the Pipeline

All scripts auto-detect the repo root, so you can run them from anywhere:

1. **Model evaluation** – generate model responses for each query split:

   ```bash
   bash scripts/02_evaluate_model.sh
   ```

   The configuration array inside the script defines which combinations of
   query type, models, and document modes are executed.  Edit the array to add
   or remove runs without touching the Python code.

2. **LLM-as-a-judge scoring** – score one or more evaluation files:

   ```bash
   bash scripts/03_judge_evaluation.sh
   ```

   The script reads the evaluation JSON files under `results/` and produces
   judged outputs under `judge_results/` plus aggregated metrics in `metrics/`.

3. **Mitigation strategies** – each script chains provider + judge stages:

   ```bash
   bash scripts/04_mitigation_cot.sh
   bash scripts/04_mitigation_dva.sh
   bash scripts/04_mitigation_revision.sh
   ```

   Each mitigation script writes intermediate generation JSON to `results/` and
   final judge outputs to `judge_results/`.  Modify their configuration arrays
   to customize models, doc-modes, or batch sizes.

All Python entry points fail fast if `OPENROUTER_API_KEY` is missing or if the
expected PDFs/JSON files are not present, which keeps accidental misconfigurations
obvious.

## Notes

- No OpenRouter keys or personal SLURM settings are stored in this folder.  The
  bash helpers simply expect you to export your own secrets.
- The copied `data/` directory contains the final benchmark instances along with
  the PDF clips they reference, so nothing external is required to re-run.
- If you only need a subset of the workflow, you can call the Python modules
  directly (e.g., `python src/02_evaluate_model.py --help`).
