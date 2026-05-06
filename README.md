# EEG Data Interpreter

Alzheimer's-focused quantitative EEG reasoning pipeline. Takes a raw EEG recording, extracts clinical biomarkers and deep embeddings via the LaBraM EEG foundation model, and produces a structured clinical report using a local LLM (Qwen3:8b via Ollama).

## Pipeline overview

```text
EEG file (any format)
    │
    ▼
preprocessor        channel validation · resampling to 200 Hz ·
                    bandpass filtering · average re-reference ·
                    artifact detection
    │
    ▼
feature_extractor   band power · alpha peak · theta/alpha ratio ·
                    posterior coherence · LaBraM embeddings ·
                    Healthy/AD/Other probabilities
    │
    ▼
reasoning_agent     Qwen3:8b via Ollama → hypotheses · evidence · confidence
    │
    ▼
output_formatter    EEGReport → human-readable text · machine-readable JSON
```

---

## Requirements

- **Python 3.10+**
- **[Ollama](https://ollama.com)** to run Qwen3:8b (or your desired reasoning llm) locally
- CUDA-capable GPU recommended for LaBraM inference

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd eeg-data-interpreter
```

### 2. Create and activate a Python environment

```bash
conda create -n eeg-data-interpreter python=3.10
conda activate eeg-data-interpreter
```

Or with venv:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

Recommended compatible versions:

```bash
pip install numpy==1.26.4 pandas==1.5.3 scipy==1.11.4 scikit-learn==1.3.2
pip install mne==1.6.1 mne-connectivity==0.6.0
```

### 4. Add the LaBraM model source

Copy the following file from your LaBraM training repository into the project root:

```text
modeling_finetune.py
```

This file contains the LaBraM architecture definition used during inference.

### 5. Add the LaBraM checkpoint

Place the trained LaBraM checkpoint at:

```text
checkpoints/labram/checkpoint-best.pth
```

This project uses a fine-tuned multiclass LaBraM model trained on:

- Healthy controls
- Alzheimer's disease
- Other neurological conditions

The model outputs subject-level probabilities for all three classes.

### 6. Install and start Ollama

Download Ollama from [ollama.com](https://ollama.com) and install it. Then pull the model:

```bash
ollama pull qwen3:8b
```

Start the Ollama server (if it isn't already running):

```bash
ollama serve
```

Leave this running in a background terminal.

### 7. Download demo data

Downloads one AD and one HC subject from the public ADFTD dataset (OpenNeuro ds004504):

```bash
python scripts/download_demo_data.py
```

Files are saved to `data/demo/AD/` and `data/demo/HC/`.

---

## Running the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

**Sidebar options:**
- **Upload** your own EEG file (EDF, BDF, FIF, SET, GDF, CNT, VHDR, CSV, NPZ, MAT)
- **Real EEG files** — the ADFTD subjects downloaded above (labelled AD / HC)
- **Synthetic test files** — generated test cases for quick sanity checks

Select a file and click **Run pipeline**. The report appears in four tabs:

| Tab | Contents |
|-----|----------|
| Objective Findings | Biomarker table, band power chart, LaBraM model stats |
| Differential Diagnosis | Ranked hypotheses with confidence and evidence |
| Uncertainty & Confidence | Limiting factors, caveats, what would change the assessment |
| Next Steps | Recommended clinical actions |

Download the full report as JSON or plain text using the buttons at the bottom.

---

## Running tests

```bash
# Unit tests (no Ollama or real data required)
python -m pytest preprocessor/tests/ feature_extractor/tests/                   reasoning_agent/tests/ output_formatter/tests/ -v

# Integration tests (requires generated test files)
python preprocessor/data/generate_test_files.py
python -m pytest preprocessor/tests/test_integration.py                   feature_extractor/tests/test_integration.py -v
```

---

## LaBraM inference setup

This project uses a fine-tuned LaBraM EEG foundation model for multiclass classification.

The inference pipeline:

1. Loads raw EEG data with MNE-Python
2. Validates and reorders EEG channels
3. Resamples recordings to 200 Hz
4. Splits the EEG into overlapping 4-second windows
5. Passes each window through LaBraM
6. Aggregates window predictions into a subject-level prediction

The model outputs:

- Healthy probability
- Alzheimer probability
- Other neurological probability

These outputs are combined with interpretable EEG biomarkers during the reasoning stage.

---

## Biomarkers used

The pipeline extracts several quantitative EEG biomarkers:

- Relative band power
  - delta
  - theta
  - alpha
  - beta

- Posterior alpha peak frequency
- Theta/alpha power ratio
- Posterior alpha coherence

These features are computed using MNE-Python and combined with LaBraM model outputs during downstream reasoning.

---

## Project structure

```text
eeg-data-interpreter/
├── preprocessor/           EEG loading, filtering, artifact detection
├── feature_extractor/      Band features, coherence, LaBraM inference
├── reasoning_agent/        Ollama/Qwen3 clinical reasoning
├── output_formatter/       Typed EEGReport, text and JSON output
├── checkpoints/
│   └── labram/
│       └── checkpoint-best.pth
├── app.py                  Streamlit frontend
├── modeling_finetune.py    LaBraM model definition
├── scripts/
│   ├── download_demo_data.py
│   └── download_labram_weights.py
├── data/                   EEG data files (gitignored)
└── requirements.txt
```

---

## Supported EEG formats

EDF, BDF, FIF, SET (EEGLAB), GDF, CNT, VHDR (BrainVision), CSV/TSV, NPZ, MAT
