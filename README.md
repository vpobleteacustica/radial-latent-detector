# Radial Latent Detector --- Quantile Sensitivity Analysis

This repository provides a minimal and fully reproducible implementation
of a radial decision rule applied in latent space.

The primary objective is to evaluate the sensitivity of classification
performance with respect to the quantile threshold parameter ( q ),
which controls the acceptance radius around each class centroid in
latent space.

The experiment systematically varies the quantile parameter and
evaluates:

-   Global Accuracy
-   Macro-F1 Score
-   Balanced Accuracy
-   No-detect (rejection) rate
-   Distance distributions per class

------------------------------------------------------------------------

## Repository Structure

    radial-latent-detector/
    │
    ├── README.md
    ├── requirements.txt
    │
    ├── scripts/
    │   ├── 20_run_q_sweep.sh
    │   ├── 21_summarize_q_sweep.py
    │   └── 22_plot_q_sweep.py
    │
    ├── latent_space_exploration/
    │   ├── 08_fit_radial_detector.py
    │   └── 10_benchmark_folder_detection.py
    │
    ├── config/
    ├── data/        # not tracked (user must provide)
    └── outputs/     # generated automatically

The directories `data/` and `outputs/` are intentionally not
version-controlled.

------------------------------------------------------------------------

## Required Data Structure

Users must provide the following directory structure inside `data/`:

``` bash
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

These files are not included in the repository.

------------------------------------------------------------------------

## Installation

``` bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Running the Quantile Sweep

``` bash
bash scripts/20_run_q_sweep.sh
```

This generates a timestamped directory inside `outputs/`, for example:

    outputs/q_sweep_YYYYMMDD_HHMMSS/

------------------------------------------------------------------------

## Summarizing Results

``` bash
python scripts/21_summarize_q_sweep.py --sweep-dir outputs/q_sweep_<TIMESTAMP>
```

This produces aggregated performance metrics for each quantile value.

------------------------------------------------------------------------

## Plotting Sensitivity Curves

``` bash
python scripts/22_plot_q_sweep.py --sweep-dir outputs/q_sweep_<TIMESTAMP>
```

Generated figures include:

-   Global accuracy vs q\
-   Macro-F1 vs q\
-   Balanced accuracy vs q\
-   Rejection (no-detect) rate vs q\
-   Distance distributions per class

------------------------------------------------------------------------

## Scientific Interpretation

Increasing the quantile threshold expands the acceptance radius in
latent space.

As ( q ) increases: - The rejection rate decreases. - The acceptance
region becomes more permissive. - Inter-class overlap tends to
increase. - Global and macro-level performance metrics may decrease.

This reveals a structural trade-off between conservative and permissive
decision regions.

The repository is intended for reproducible experimentation and
quantitative sensitivity analysis of latent-space radial decision rules.
