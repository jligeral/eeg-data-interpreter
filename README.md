# EEG Data Interpreter

Alzheimer's-focused quantitative EEG reasoning pipeline. Takes a raw EEG recording, extracts clinical biomarkers and deep embeddings via the LEAD foundation model, and produces a structured clinical report using a local LLM (Qwen3:8b via Ollama).

## Pipeline overview

```
EEG file (any format)
    │
    ▼
preprocessor        bandpass filter · average re-reference · artifact detection
    │
    ▼
feature_extractor   band power · alpha peak · theta/alpha ratio · posterior coherence
                    LEAD embeddings (2432-dim) · AD probability
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
- **[Ollama](https://ollama.com)** — runs Qwen3:8b locally
- **16 GB RAM** recommended (Qwen3:8b is 5.2 GB; LEAD model adds ~200 MB)
- Apple Silicon (MPS) or CPU — no GPU required

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd eeg-data-interpreter
```

### 2. Create and activate a Python environment

```bash
conda create -n eeg-data-interpreter python=3.12
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

### 4. Clone the LEAD model source

LEAD is a third-party EEG foundation model. Clone it into the `lead/` directory:

```bash
git clone https://github.com/DL4mHealth/LEAD lead
```

### 5. Get the LEAD checkpoint

Place the fine-tuned checkpoint at:

```
checkpoints/LEADv2/finetuned/checkpoint.pth
```

Ask a teammate who has already run the fine-tuning for this file, or follow the **Fine-tuning** section below to produce it yourself.

> If you only have the P-Base pretrain checkpoint, place it at `checkpoints/LEADv2/P-Base/checkpoint.pth` and update `_CHECKPOINT` in `feature_extractor/lead_encoder.py` accordingly. AD probability will not be available with the pretrain checkpoint.

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

### 7. Download real demo data (optional)

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
| Objective Findings | Biomarker table, band power chart, LEAD embedding stats |
| Differential Diagnosis | Ranked hypotheses with confidence and evidence |
| Uncertainty & Confidence | Limiting factors, caveats, what would change the assessment |
| Next Steps | Recommended clinical actions |

Download the full report as JSON or plain text using the buttons at the bottom.

---

## Running tests

```bash
# Unit tests (no Ollama or real data required)
python -m pytest preprocessor/tests/ feature_extractor/tests/ \
                  reasoning_agent/tests/ output_formatter/tests/ -v

# Integration tests (requires generated test files)
python preprocessor/data/generate_test_files.py
python -m pytest preprocessor/tests/test_integration.py \
                  feature_extractor/tests/test_integration.py -v
```

---

## Fine-tuning LEAD (to get AD probability)

The AD probability score requires a fine-tuned LEAD checkpoint trained on labeled AD/HC data. The pretrain checkpoint only produces embeddings.

### Option A — Use Google Colab (recommended)

1. Open a new Colab notebook with a **T4 or A100 GPU** runtime
2. Run the following cells:

```python
# Install dependencies
!git clone https://github.com/DL4mHealth/LEAD
%cd LEAD
!pip install -q --no-deps "reformer-pytorch==1.4.4" "linear-attention-transformer==0.19.1"
!pip install -q "timm==0.6.13" "natsort" "gdown" "mne"

# Fix compatibility issues with newer Python/NumPy/PyTorch
!sed -i 's/super().__init__(dataset)/super().__init__()/' utils/tools.py
!sed -i 's/np\.Inf/np.inf/g' utils/tools.py

# Download P-Base pretrain checkpoint
!mkdir -p checkpoints/LEADv2/pretrain_lead/LEADv2/P-Base/nh8_el12_dm128_df256_seed41
!gdown --folder https://drive.google.com/drive/folders/1_XUfU3vZB40rjivkNYf8L2slCahXPo43 \
    -O checkpoints/LEADv2/pretrain_lead/LEADv2/P-Base/nh8_el12_dm128_df256_seed41/
# Fix doubled path if needed:
!find checkpoints -name "checkpoint.pth" | head -5

# Download ADFTD fine-tuning dataset
!mkdir -p dataset/L400
!gdown --folder https://drive.google.com/drive/folders/1y66f_Id-kal7q8uu-YYF2qTUHfhbPXOX \
    -O dataset/L400/

# Fine-tune
!python -u run.py --method LEADv2 \
    --checkpoints_path ./checkpoints/LEADv2/pretrain_lead/LEADv2/P-Base/nh8_el12_dm128_df256_seed41/checkpoint.pth \
    --task_name finetune --is_training 1 \
    --root_path ./dataset/L400/dataset/L400/ \
    --model_id P-Base-F-ADFTD --model LEADv2 --data MultiDatasets \
    --training_datasets ADFTD --testing_datasets ADFTD \
    --e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 \
    --patch_len 50 --stride 50 --group_shuffle --group_size 8 \
    --sampling_rate_list 200,100,50 --ratio_a 0.8 --ratio_b 0.9 \
    --montage_name standard_1005 \
    --channel_names Fp1,Fp2,F7,F3,Fz,F4,F8,T7,C3,Cz,C4,T8,P7,P3,Pz,P4,P8,O1,O2 \
    --classify_choice ad_vs_hc --cross_val mccv \
    --learning_rate 0.0001 --train_epochs 50 --patience 10

# Save checkpoint to Google Drive
from google.colab import drive
drive.mount('/content/drive')
import shutil, glob
ckpt = glob.glob('checkpoints/LEADv2/finetune/**/*.pth', recursive=True)[0]
shutil.copy(ckpt, '/content/drive/MyDrive/lead_finetuned_checkpoint.pth')
print('Saved:', ckpt)
```

3. Download `lead_finetuned_checkpoint.pth` and place it at `checkpoints/LEADv2/finetuned/checkpoint.pth` in this project.

> Training takes ~4 hours on T4, ~45 minutes on A100.

### Option B — Use your own labeled data

If you have EEG recordings labeled as AD or HC, organise them as:

```
data/
  AD/   ← EEG files for Alzheimer's patients
  HC/   ← EEG files for healthy controls
```

Then run:

```bash
python scripts/prepare_lead_dataset.py \
    --ad_dir data/AD \
    --hc_dir data/HC \
    --name   MyDataset
```

This converts your data into LEAD's format. Follow the printed fine-tuning command to train.

---

## Project structure

```
eeg-data-interpreter/
├── preprocessor/           EEG loading, filtering, artifact detection
├── feature_extractor/      Band features, coherence, LEAD embeddings
├── reasoning_agent/        Ollama/Qwen3 clinical reasoning
├── output_formatter/       Typed EEGReport, text and JSON output
├── app.py                  Streamlit frontend
├── scripts/
│   ├── download_demo_data.py     Fetch real AD/HC EEG files
│   ├── prepare_lead_dataset.py   Convert labeled EEGs for LEAD fine-tuning
│   └── download_lead_weights.py  Fetch P-Base pretrain checkpoint
├── lead/                   LEAD source (cloned separately, gitignored)
├── checkpoints/            Model weights (gitignored)
└── data/                   EEG data files (gitignored)
```

---

## Supported EEG formats

EDF, BDF, FIF, SET (EEGLAB), GDF, CNT, VHDR (BrainVision), CSV/TSV, NPZ, MAT
