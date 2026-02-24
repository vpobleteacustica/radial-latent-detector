# Radial Latent Detector — Quantile Sensitivity Analysis

This repository provides a minimal, reproducible implementation of a radial decision rule applied in latent space. The main objective is to evaluate the sensitivity of classification performance with respect to the quantile threshold parameter \( q \).

The experiment systematically varies the quantile parameter and evaluates:

- Global Accuracy
- Macro-F1 Score
- Balanced Accuracy
- No-detect (rejection) rate
- Distance distributions per class

---

## Repository Structure
```bash
data/
├── downloaded_models/
│   └── bird_net_vae_audio_splitted_encoder_v0/
│       ├── model.pt
│       └── bird_net_vae_audio_splitted.yaml
│
├── train_chunks/
├── val_chunks/
└── test_chunks/
```
These directories are not included in the repository.


## Required Data Structure

Users must provide the following directory structure inside `data/`:

```bash
data/
├── downloaded_models/
│   └── bird_net_vae_audio_splitted_encoder_v0/
│       ├── model.pt
│       └── bird_net_vae_audio_splitted.yaml
│
├── train_chunks/
├── val_chunks/
└── test_chunks/
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Quantile Sweep

```bash
bash scripts/20_run_q_sweep.sh
```

This will generate a timestamped directory inside outputs/.

## Summarizing Results

```bash
python scripts/21_summarize_q_sweep.py --sweep-dir outputs/q_sweep_<TIMESTAMP>
```

## Plotting Sensitivity Curves

```bash
python scripts/22_plot_q_sweep.py --sweep-dir outputs/q_sweep_<TIMESTAMP>
```

Generated figures include:
	•	Global accuracy vs q
	•	Macro-F1 vs q
	•	Rejection rate vs q
	•	Distance distributions per class

## Scientific Interpretation

Increasing the quantile threshold expands the acceptance radius in latent space, reducing the rejection rate but increasing inter-class overlap. This reveals a structural trade-off between conservative and permissive decision regions.

The repository is intended for reproducible experimentation and sensitivity analysis.

---

Luego guarda y ejecuta:

```bash
git add README.md
git commit -m "Add README with reproducibility instructions"
git push




