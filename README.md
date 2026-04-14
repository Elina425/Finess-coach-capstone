# Fitness coach capstone (code)

Python pipeline for **pose estimation → keypoint preprocessing → biomechanical angles → sequence models (BiLSTM / GCN)**.

**Primary data path:** [Riccio et al.–style Kaggle exercise videos](https://www.kaggle.com/datasets) processed through this repo’s Riccio pipeline (steps 3–5). The capstone runbook does **not** assume the legacy QEVD-FIT-COACH layout.

## Run order (steps 2–5)

Work through the pipeline in order; each stage depends on the previous one.

| Step | What | Where to start |
|------|------|----------------|
| **2** | Compare pose models (speed, skeleton quality) | [`pipeline/step_02_pose_comparison.sh`](pipeline/step_02_pose_comparison.sh) → `run_full_comparison.py` |
| **3** | Normalize, impute, sync FPS | Bundled in step 4 script for Riccio (`step_03_04_riccio_videos_to_angles.sh`) |
| **4** | Joint angles from keypoints | Same Riccio script → `*_biomechanics.npz` under e.g. `results/riccio_realtime_exercise_recognition/` |
| **5** | BiLSTM (30-frame windows) | [`pipeline/step_05_train_bilstm_riccio.sh`](pipeline/step_05_train_bilstm_riccio.sh) → `train_exercise_bilstm.py --kaggle-angles-dir …` |

**Riccio step-by-step (tasks 2–5):** [`pipeline/README.md`](pipeline/README.md) — setup, dataset acquisition, steps 2 → 5, train/val/test split (~70% / 15% / 15% by video), epochs and `eval-test` outputs, optional evaluation / Colab / EgoExo.

**Deep dive:** [`docs/CAPSTONE_PIPELINE.md`](docs/CAPSTONE_PIPELINE.md) (code ↔ thesis steps), [`docs/README.md`](docs/README.md) (doc index).

**Google Colab:** [`notebooks/colab_gpu_training.ipynb`](notebooks/colab_gpu_training.ipynb) — GPU check, install, BiLSTM on prebuilt Riccio NPZs.

**EgoExo (optional):** human quality labels + frames — short pointer in [`pipeline/README.md`](pipeline/README.md) §6 and `pipeline/step_06_train_bilstm_egoexo.sh`.

## Setup

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install -e .
```

Editable install (`pip install -e .`) registers the `fitness_coach` package so imports and the root script shims work. If you skip it, set `PYTHONPATH` to the repo root.

## Project layout

| Location | Role |
|----------|------|
| [`pipeline/`](pipeline/) | **Ordered runbook** (README + step scripts 2 → 5 for Riccio) |
| [`fitness_coach/core/`](fitness_coach/core/) | Pose estimation, biomechanics, labels, KNN imputation |
| [`fitness_coach/datasets/`](fitness_coach/datasets/) | PyTorch datasets (BiLSTM windows, ST-GCN, GCN keypoint windows) |
| [`fitness_coach/models/`](fitness_coach/models/) | BiLSTM, ST-GCN, GCN pose heads / topology |
| [`fitness_coach/pipelines/`](fitness_coach/pipelines/) | Riccio / Kaggle video preprocessing, batch angles, comparisons (optional legacy dataset helpers) |
| [`fitness_coach/training/`](fitness_coach/training/) | Train/tune scripts, index building, stratified splits |
| [`fitness_coach/evaluation/`](fitness_coach/evaluation/) | Model and pose metrics, confusion / ROC plots |
| [`fitness_coach/inference/`](fitness_coach/inference/) | BiLSTM and GCN-similarity inference |
| [`fitness_coach/utils/`](fitness_coach/utils/) | IMU features, KNN batch, downloads, visualizations |
| Repo root `*.py` | Thin shims: e.g. `train_exercise_bilstm.py` → `fitness_coach.training.train_exercise_bilstm` |
| [`scripts/`](scripts/) | Shell helpers (e.g. YouTube download) |
| [`docs/`](docs/) | Capstone write-ups and figures |

**Common entry points:** `keypoint_preprocessing_pipeline.py`, `riccio_kaggle_video_pipeline.py`, `train_exercise_bilstm.py`, `run_full_comparison.py`, `quickstart.py`, `run_pipeline.py`.

Equivalent module form: `python -m fitness_coach.training.train_exercise_bilstm --help`.
