# Doc-PP: Document Policy Preservation Benchmark

<p align="center">
  <img src="assets/fig_doc-pp_intro.png" width="800"/>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2601.03926"><img src="https://img.shields.io/badge/arXiv-2601.03926-b31b1b.svg" alt="arXiv"></a>
</p>

We introduce **Doc-PP**, a benchmark for evaluating whether Large Vision-Language Models (LVLMs) can adhere to user-defined non-disclosure policies when answering questions about multimodal documents. Our evaluation reveals a *Reasoning-Induced Safety Gap*: models frequently leak sensitive information when complex reasoning or cross-modal synthesis is required. To address this, we propose **DVA (Decompose–Verify–Aggregation)**, a structural inference framework that decouples reasoning from policy verification and significantly outperforms standard prompting defenses.

## Repository Layout

- `data.zip`, `data.z01` ~ `data.z06` – split compressed archives containing the benchmark JSON and PDFs. See [Data Setup](#data-setup).
- `prompts/` – system/user prompt templates referenced by the Python code.
- `src/` – async OpenRouter evaluation, judging, and mitigation drivers.
- `scripts/` – bash wrappers for the submission pipeline.

## Data Setup

The data is split into multiple compressed archives due to GitHub file size limits. Extract them before running:

```bash
zip -s 0 data.zip --out data_combined.zip
unzip data_combined.zip
rm data_combined.zip  # optional
```

This creates the `data/` directory:
```
data/
├── 02_final_faithfulness_checklists.json   # benchmark JSON
└── docs_clip/                               # PDF documents
    └── *.pdf
```

## Environment Setup

1. Use Python 3.10+.
2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Export your OpenRouter credentials:

   ```bash
   export OPENROUTER_API_KEY="sk-or-your-key"
   ```

## Running the Pipeline

1. **Model evaluation** – generate model responses:
   ```bash
   bash scripts/02_evaluate_model.sh
   ```

2. **LLM-as-a-judge scoring** – score evaluation files:
   ```bash
   bash scripts/03_judge_evaluation.sh
   ```

3. **Mitigation strategies** – run DVA and other defenses:
   ```bash
   bash scripts/04_mitigation_cot.sh
   bash scripts/04_mitigation_dva.sh
   bash scripts/04_mitigation_revision.sh
   ```

## Acknowledgments

The documents used in this benchmark are based on:

- [MMLongBench-Doc](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ae0e43289bffea0c1fa34633fc608e92-Abstract-Datasets_and_Benchmarks_Track.html)
- [SustainableQA](https://arxiv.org/abs/2508.03000)

We sincerely thank the authors for making their datasets available.
