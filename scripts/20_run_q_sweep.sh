#!/usr/bin/env bash
set -euo pipefail

# Uso:
#   bash scripts/20_run_q_sweep.sh
# Opcional:
#   DEVICE=cpu bash scripts/20_run_q_sweep.sh
#   QS="0.95 0.97 0.98 0.99 0.995 0.999" bash scripts/20_run_q_sweep.sh

export WANDB_MODE=${WANDB_MODE:-offline}
DEVICE=${DEVICE:-cpu}
QS=${QS:-"0.95 0.97 0.98 0.99 0.995 0.999"}

ENC_PT="data/downloaded_models/bird_net_vae_audio_splitted_encoder_v0/model.pt"
ENC_YAML="data/downloaded_models/bird_net_vae_audio_splitted_encoder_v0/bird_net_vae_audio_splitted.yaml"

TRAIN_ROOT="data/train_chunks"
VAL_ROOT="data/val_chunks"
TEST_ROOT="data/test_chunks"

STAMP=$(date +"%Y%m%d_%H%M%S")
OUTROOT="outputs/q_sweep_${STAMP}"
mkdir -p "$OUTROOT"

echo "== Q SWEEP =="
echo "OUTROOT: $OUTROOT"
echo "DEVICE:  $DEVICE"
echo "QS:      $QS"
echo

for q in $QS; do
  qtag=$(python - <<PY
q=float("$q")
print(f"{q:.3f}")
PY
)
  qdir="$OUTROOT/q_${qtag}"
  mkdir -p "$qdir/train" "$qdir/val" "$qdir/test"

  echo "=============================================="
  echo "▶️  q=$qtag  (fit radial detector -> config.json)"
  echo "=============================================="

  python latent_space_exploration/08_fit_radial_detector.py \
    --encoder-pt "$ENC_PT" \
    --encoder-yaml "$ENC_YAML" \
    --device "$DEVICE" \
    --q "$q"

  # Snapshot del config usado (importante para reproducibilidad)
  cp -f "config.json" "$qdir/config.snapshot.json"

  echo
  echo "▶️  Benchmark TRAIN"
  python latent_space_exploration/10_benchmark_folder_detection.py \
    --root "$TRAIN_ROOT" \
    --config "config.json" \
    --device "$DEVICE"
  # Los outputs por defecto van a outputs/detection_benchmark/* (según el script).
  # Los copiamos a nuestra carpeta del sweep:
  cp -f outputs/detection_benchmark/results.csv "$qdir/train/results.csv"
  cp -f outputs/detection_benchmark/summary.txt "$qdir/train/summary.txt"
  cp -f outputs/detection_benchmark/*.png "$qdir/train/" 2>/dev/null || true

  echo
  echo "▶️  Benchmark VAL"
  python latent_space_exploration/10_benchmark_folder_detection.py \X
    --root "$VAL_ROOT" \
    --config "config.json" \
    --device "$DEVICE"
  cp -f outputs/detection_benchmark/results.csv "$qdir/val/results.csv"
  cp -f outputs/detection_benchmark/summary.txt "$qdir/val/summary.txt"
  cp -f outputs/detection_benchmark/*.png "$qdir/val/" 2>/dev/null || true

  echo
  echo "▶️  Benchmark TEST"
  python latent_space_exploration/10_benchmark_folder_detection.py \
    --root "$TEST_ROOT" \
    --config "config.json" \
    --device "$DEVICE"
  cp -f outputs/detection_benchmark/results.csv "$qdir/test/results.csv"
  cp -f outputs/detection_benchmark/summary.txt "$qdir/test/summary.txt"
  cp -f outputs/detection_benchmark/*.png "$qdir/test/" 2>/dev/null || true

  echo
  echo "✅ Finished q=$qtag -> $qdir"
  echo
done

echo "✅ Sweep done: $OUTROOT"
echo "Next: python scripts/21_summarize_q_sweep.py --sweep-dir $OUTROOT"
