# %% [markdown]
# # Fitness Coach Capstone - Complete Pipeline for Google Colab
# 
# This notebook executes the complete capstone pipeline:
# - **Step 0**: Setup and environment configuration
# - **Step 1**: Mount Google Drive and clone repository
# - **Step 2**: Pose backend comparison (optional)
# - **Step 3-4**: Process Riccio videos → extract angles, keypoints, and labels (NPZ format)
# - **Step 5**: Train BiLSTM model on extracted angles for exercise classification
# 
# **Total Runtime**: ~1-3 hours depending on dataset size and GPU availability.

# %% [markdown]
# ## Step 0: Environment Setup for Colab

# %%
# Check if running in Colab
import sys
try:
    from google.colab import drive
    IN_COLAB = True
    print("✓ Running in Google Colab")
except ImportError:
    IN_COLAB = False
    print("⚠ Not running in Colab - some features may not work as expected")

# Check GPU availability
import torch
print(f"\n GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f" GPU Name: {torch.cuda.get_device_name(0)}")
    print(f" GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## Step 1: Mount Google Drive and Clone Repository

# %%
import os
from pathlib import Path

if IN_COLAB:
    # Mount Google Drive
    from google.colab import drive
    drive.mount('/content/drive')
    print("✓ Google Drive mounted at /content/drive")
    
    # Define workspace path
    WORKSPACE_ROOT = Path("/content/drive/My Drive/Fitness-Coach-Capstone")
else:
    # Local development
    WORKSPACE_ROOT = Path.cwd()

print(f"\n Workspace root: {WORKSPACE_ROOT}")
print(f" Exists: {WORKSPACE_ROOT.exists()}")

# %%
# Install dependencies if needed
import subprocess

os.chdir(WORKSPACE_ROOT)

# Install requirements
print("Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)

# Optional: Install kagglehub for automatic dataset download
try:
    import kagglehub
    print("✓ kagglehub already installed")
except ImportError:
    print("Installing kagglehub for automatic dataset download...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"], check=True)

print("\n✓ Dependencies installed successfully")

# %% [markdown]
# ## Step 2: Pose Backend Comparison (Optional - Skip for Speed)
# 
# This step compares different pose detectors (MediaPipe, OpenPose, etc.). You can skip this and go directly to Step 3-4 for faster pipeline execution.

# %%
# Configuration for Step 2 (Pose Comparison)
SKIP_POSE_COMPARISON = True  # Set to False to run pose comparison
NUM_VIDEOS_COMPARISON = 5    # Number of videos to analyze
OUTPUT_DIR_COMPARISON = WORKSPACE_ROOT / "results/02_pose_comparison"

if not SKIP_POSE_COMPARISON:
    print(f"Running pose backend comparison on {NUM_VIDEOS_COMPARISON} videos...")
    from fitness_coach.pipelines.run_full_comparison import main as compare_main
    
    # Prepare arguments
    comparison_args = [
        "--num-videos", str(NUM_VIDEOS_COMPARISON),
        "--output-dir", str(OUTPUT_DIR_COMPARISON),
    ]
    
    # Run comparison
    sys.argv = ["run_full_comparison.py"] + comparison_args
    try:
        compare_main()
        print("✓ Pose comparison completed")
    except Exception as e:
        print(f"✗ Pose comparison failed: {e}")
else:
    print("⊘ Pose comparison skipped (SKIP_POSE_COMPARISON=True)")

# %% [markdown]
# ## Step 3-4: Process Riccio Videos → Extract Angles, Keypoints, and Labels
# 
# This step:
# 1. Downloads the Riccio dataset from Kaggle (if not already cached)
# 2. Extracts MediaPipe keypoints from each frame
# 3. Applies preprocessing (normalization, imputation, FPS sync)
# 4. Computes biomechanical angles and features
# 5. Generates NPZ files for model training

# %%
# Configuration for Steps 3-4 (Video Processing)
import os

# Output paths
RICCIO_OUTPUT_DIR = WORKSPACE_ROOT / "results/riccio_realtime_exercise_recognition"
RICCIO_STEM = "riccio_realtime_exercise_recognition"

# Processing parameters
MAX_VIDEOS = 10  # Reduce for faster processing during testing; use 0 for all
WORKERS = 0      # 0 = auto-detect CPU count; set to specific number to limit parallelism
TARGET_FPS = 30  # Resample video to this FPS

# Preprocessing techniques (select your setup)
USE_RICH_PREPROCESSING = False  # Use full preprocessing stack

# Preprocessing flags
SKIP_KEYPOINTS = False          # Set to True to skip large keypoint NPZ files
DOWNLOAD_DATASET = True         # Set to True to auto-download from Kaggle

# Create output directory
RICCIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set environment variables
os.environ["RICCIO_OUTPUT_DIR"] = str(RICCIO_OUTPUT_DIR)
os.environ["RICCIO_STEM"] = RICCIO_STEM
os.environ["RICCIO_MP_MAX_WORKERS"] = "8"

print(f"Output directory: {RICCIO_OUTPUT_DIR}")
print(f"Output stem: {RICCIO_STEM}")
print(f"Max videos: {MAX_VIDEOS if MAX_VIDEOS > 0 else 'All'}")
print(f"Preprocessing: {'Rich (full stack)' if USE_RICH_PREPROCESSING else 'Standard (norm, impute, FPS)'}")

# %%
# Run video processing pipeline (Steps 3-4)
print("\n" + "="*70)
print("STEP 3-4: Processing Riccio Videos → NPZ Format")
print("="*70 + "\n")

from fitness_coach.pipelines.riccio_kaggle_video_pipeline import main as video_pipeline_main

# Build arguments for video pipeline
video_args = [
    "--output-dir", str(RICCIO_OUTPUT_DIR),
    "--output-stem", RICCIO_STEM,
    "--target-fps", str(TARGET_FPS),
    "--workers", str(WORKERS),
]

# Add optional flags
if MAX_VIDEOS > 0:
    video_args.extend(["--max-videos", str(MAX_VIDEOS)])

if DOWNLOAD_DATASET:
    video_args.append("--download")

if SKIP_KEYPOINTS:
    video_args.append("--skip-keypoints")

if USE_RICH_PREPROCESSING:
    video_args.append("--rich-preprocess")

print(f"Arguments: {' '.join(video_args)}\n")

# Run pipeline
sys.argv = ["riccio_kaggle_video_pipeline.py"] + video_args
try:
    video_pipeline_main()
    print("\n✓ Video processing completed successfully")
except Exception as e:
    print(f"\n✗ Video processing failed: {e}")
    import traceback
    traceback.print_exc()

# %%
# Verify output files
import glob

print("\nVerifying output files...\n")

npz_files = {
    "Biomechanics": glob.glob(str(RICCIO_OUTPUT_DIR / f"*_biomechanics.npz")),
    "Labels": glob.glob(str(RICCIO_OUTPUT_DIR / f"*_labels.npz")),
    "Keypoints": glob.glob(str(RICCIO_OUTPUT_DIR / f"*_keypoints.npz")),
}

for file_type, files in npz_files.items():
    if files:
        print(f"✓ {file_type}: {len(files)} file(s)")
        for f in files[:3]:  # Show first 3
            size_mb = Path(f).stat().st_size / (1024**2)
            print(f"  - {Path(f).name} ({size_mb:.1f} MB)")
    else:
        print(f"✗ {file_type}: No files found")

# %% [markdown]
# ## Step 5: Train BiLSTM Model on Exercise Classification
# 
# Train a BiLSTM model on 30-frame windows of biomechanical angles for exercise classification and quality regression.

# %%
# Configuration for Step 5 (BiLSTM Training)

# Training parameters
BILSTM_OUTPUT_DIR = WORKSPACE_ROOT / "results/exercise_bilstm"
BILSTM_EPOCHS = 30
BILSTM_BATCH_SIZE = 32
BILSTM_LEARNING_RATE = 0.001

# Architecture and training mode
BILSTM_ARCHITECTURE = "cnn_attn"  # Options: baseline, cnn_attn
CLASSIFICATION_ONLY = True         # True = classification only; False = classification + regression
STANDARDIZE_FEATURES = True        # Standardize input features
EVAL_ON_TEST = True                # Evaluate on test set after training

# Data split ratios
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Create output directory
BILSTM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Training output directory: {BILSTM_OUTPUT_DIR}")
print(f"Epochs: {BILSTM_EPOCHS}")
print(f"Batch size: {BILSTM_BATCH_SIZE}")
print(f"Learning rate: {BILSTM_LEARNING_RATE}")
print(f"Architecture: {BILSTM_ARCHITECTURE}")
print(f"Classification only: {CLASSIFICATION_ONLY}")
print(f"Standardize: {STANDARDIZE_FEATURES}")

# %%
# Run BiLSTM training (Step 5)
print("\n" + "="*70)
print("STEP 5: Training BiLSTM Model")
print("="*70 + "\n")

from fitness_coach.training.train_exercise_bilstm import main as train_bilstm_main

# Build arguments for BiLSTM training
train_args = [
    "--preset", "riccio",
    "--kaggle-angles-dir", str(RICCIO_OUTPUT_DIR),
    "--kaggle-stem", RICCIO_STEM,
    "--output-dir", str(BILSTM_OUTPUT_DIR),
    "--epochs", str(BILSTM_EPOCHS),
    "--batch-size", str(BILSTM_BATCH_SIZE),
    "--learning-rate", str(BILSTM_LEARNING_RATE),
    "--architecture", BILSTM_ARCHITECTURE,
    "--kaggle-val-ratio", str(VAL_RATIO),
    "--kaggle-test-ratio", str(TEST_RATIO),
]

# Add optional flags
if STANDARDIZE_FEATURES:
    train_args.append("--standardize")

if CLASSIFICATION_ONLY:
    train_args.append("--classification-only")

if EVAL_ON_TEST:
    train_args.append("--eval-test")

print(f"Arguments: {' '.join(train_args)}\n")

# Run training
sys.argv = ["train_exercise_bilstm.py"] + train_args
try:
    train_bilstm_main()
    print("\n✓ BiLSTM training completed successfully")
except Exception as e:
    print(f"\n✗ BiLSTM training failed: {e}")
    import traceback
    traceback.print_exc()

# %% [markdown]
# ## Summary: Results and Artifacts

# %%
# Display summary of generated artifacts
import json
from pathlib import Path

print("\n" + "="*70)
print("PIPELINE SUMMARY")
print("="*70 + "\n")

# Check preprocessing outputs
print("📊 PREPROCESSING OUTPUTS (Step 3-4)")
print("-" * 70)
preprocess_dir = RICCIO_OUTPUT_DIR
if preprocess_dir.exists():
    npz_files = list(preprocess_dir.glob("*.npz"))
    print(f"Generated {len(npz_files)} NPZ files:")
    total_size = 0
    for f in sorted(npz_files):
        size_mb = f.stat().st_size / (1024**2)
        total_size += size_mb
        print(f"  • {f.name}: {size_mb:.1f} MB")
    print(f"\nTotal preprocessing data: {total_size:.1f} MB")
else:
    print(f"Directory not found: {preprocess_dir}")

print("\n")

# Check training outputs
print("🎯 TRAINING OUTPUTS (Step 5)")
print("-" * 70)
training_dir = BILSTM_OUTPUT_DIR
if training_dir.exists():
    # List checkpoints
    checkpoints = list(training_dir.glob("*.pth"))
    if checkpoints:
        print(f"Model checkpoints: {len(checkpoints)}")
        for cp in sorted(checkpoints):
            size_mb = cp.stat().st_size / (1024**2)
            print(f"  • {cp.name}: {size_mb:.1f} MB")
    
    # List metrics
    metrics_file = training_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        print(f"\nTraining metrics:")
        if "best_val_acc" in metrics:
            print(f"  • Best validation accuracy: {metrics['best_val_acc']:.4f}")
        if "test_acc" in metrics:
            print(f"  • Test accuracy: {metrics['test_acc']:.4f}")
        if "epochs_trained" in metrics:
            print(f"  • Epochs trained: {metrics['epochs_trained']}")
    
    # List logs
    logs = list(training_dir.glob("*.log"))
    if logs:
        print(f"\nLog files: {len(logs)}")
        for log in sorted(logs):
            print(f"  • {log.name}")
else:
    print(f"Directory not found: {training_dir}")

print("\n" + "="*70)
print("✓ Pipeline execution complete!")
print("="*70)

# %% [markdown]
# ## Optional: Download Results from Google Drive
# 
# If running in Colab, you can download the results to Google Drive.

# %%
if IN_COLAB:
    import shutil
    
    drive_results = Path("/content/drive/My Drive/Fitness-Coach-Results")
    drive_results.mkdir(exist_ok=True)
    
    # Copy results
    results_local = WORKSPACE_ROOT / "results"
    if results_local.exists():
        print(f"Copying results to Google Drive...")
        
        for source_dir in results_local.iterdir():
            if source_dir.is_dir():
                dest = drive_results / source_dir.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(source_dir, dest)
                print(f"  ✓ Copied {source_dir.name}")
        
        print(f"\nResults available at: {drive_results}")
else:
    print("Not running in Colab - results are already in the workspace.")
    print(f"Results location: {WORKSPACE_ROOT / 'results'}")


