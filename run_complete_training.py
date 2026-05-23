#!/usr/bin/env python3
"""
Fitness Coach Capstone: Complete Multi-Stage Training Pipeline

This script runs a comprehensive 4-stage training pipeline:
  1. Quick Validation (15 epochs) - Verify setup works
  2. Baseline Model (50 epochs) - Establish performance
  3. Optimized Model (100 epochs) - Improve convergence
  4. Model Comparison & Export - Select best model

Usage:
  chmod +x run_complete_training.py
  ./venv/bin/python run_complete_training.py
"""

import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime


class TrainingPipeline:
    """Multi-stage training pipeline manager."""
    
    def __init__(self, workspace_root=None):
        self.workspace = Path(workspace_root or Path.cwd())
        self.results_dir = self.workspace / "results"
        self.riccio_dir = self.results_dir / "riccio_realtime_exercise_recognition"
        self.riccio_stem = "riccio_realtime_exercise_recognition"
        self.results = {}
        self.start_time = datetime.now()
        
    def validate_prerequisites(self):
        """Verify all required data files exist."""
        print("\n" + "="*80)
        print("VALIDATING PREREQUISITES")
        print("="*80)
        
        required_files = [
            self.riccio_dir / f"{self.riccio_stem}_biomechanics.npz",
            self.riccio_dir / f"{self.riccio_stem}_keypoints.npz",
            self.riccio_dir / f"{self.riccio_stem}_labels.npz",
        ]
        
        all_exist = True
        for f in required_files:
            exists = f.exists()
            status = "✓" if exists else "✗"
            print(f"{status} {f.name}")
            all_exist = all_exist and exists
        
        if not all_exist:
            print("\n✗ Missing required data files!")
            return False
        
        print("\n✅ All prerequisites satisfied")
        return True
    
    def run_stage(self, stage_name, stage_num, output_dir, config):
        """Run a single training stage."""
        print("\n" + "="*80)
        print(f"STAGE {stage_num}: {stage_name}")
        print("="*80)
        
        output_path = self.results_dir / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = [
            "./venv/bin/python",
            "train_exercise_bilstm.py",
            "--preset", "riccio",
            "--standardize",
            "--eval-test",
            "--epochs", str(config["epochs"]),
            "--batch-size", str(config["batch_size"]),
            "--lr", str(config["lr"]),
            "--kaggle-angles-dir", str(self.riccio_dir),
            "--kaggle-stem", self.riccio_stem,
            "--output-dir", str(output_path),
        ]
        
        print(f"\nConfiguration:")
        print(f"  Epochs: {config['epochs']}")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Learning Rate: {config['lr']}")
        print(f"  Output: {output_dir}")
        print(f"\n📊 Training Progress:")
        
        # Run training
        stage_start = time.time()
        result = subprocess.run(cmd, cwd=str(self.workspace))
        stage_duration = time.time() - stage_start
        
        if result.returncode != 0:
            print(f"\n✗ Stage {stage_num} failed! Duration: {stage_duration/60:.1f} min")
            return False
        
        # Load metrics
        metrics_file = output_path / "test_classification_metrics.json"
        stage_results = {
            "duration_minutes": stage_duration / 60,
            "status": "success"
        }
        
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
                stage_results.update({
                    "test_accuracy": metrics.get("test_accuracy", 0),
                    "f1_macro": metrics.get("f1_macro", 0),
                    "quality_rmse": metrics.get("test_quality_rmse", 0),
                })
        
        self.results[output_dir] = stage_results
        
        # Print stage summary
        print(f"\n✅ Stage {stage_num} Complete!")
        print(f"   Duration: {stage_duration/60:.1f} minutes")
        print(f"   Test Accuracy: {stage_results.get('test_accuracy', 'N/A'):.4f}")
        print(f"   F1 (macro): {stage_results.get('f1_macro', 'N/A'):.4f}")
        print(f"   Quality RMSE: {stage_results.get('quality_rmse', 'N/A'):.4f}")
        
        return True
    
    def compare_models(self):
        """Compare results across all training stages."""
        print("\n" + "="*80)
        print("MODEL COMPARISON & RESULTS")
        print("="*80)
        
        if not self.results:
            print("No results to compare")
            return
        
        print("\nResults Summary:")
        print("-" * 80)
        print(f"{'Model':<40} {'Accuracy':<12} {'F1 (macro)':<12} {'Time (min)':<12}")
        print("-" * 80)
        
        best_model = None
        best_accuracy = 0
        
        for model_name, metrics in self.results.items():
            accuracy = metrics.get("test_accuracy", 0)
            f1 = metrics.get("f1_macro", 0)
            duration = metrics.get("duration_minutes", 0)
            
            marker = " ⭐" if accuracy > best_accuracy else ""
            print(f"{model_name:<40} {accuracy:<12.4f} {f1:<12.4f} {duration:<12.1f}{marker}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name
        
        print("-" * 80)
        
        if best_model:
            print(f"\n🏆 Best Model: {best_model}")
            print(f"   Accuracy: {self.results[best_model]['test_accuracy']:.4f}")
            best_checkpoint = (
                self.results_dir / best_model / "exercise_bilstm_best.pt"
            )
            if best_checkpoint.exists():
                print(f"   Checkpoint: {best_checkpoint}")
        
        return best_model
    
    def run_all_stages(self):
        """Execute the complete pipeline."""
        print("\n" + "="*80)
        print("FITNESS COACH CAPSTONE: COMPLETE TRAINING PIPELINE")
        print("="*80)
        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Workspace: {self.workspace}")
        
        # Validate
        if not self.validate_prerequisites():
            return False
        
        # Stage 1: Quick Validation
        if not self.run_stage(
            "Quick Validation",
            1,
            "exercise_bilstm_stage1_validation",
            {"epochs": 15, "batch_size": 54, "lr": 0.001}
        ):
            print("\n⚠️ Stage 1 failed. Aborting.")
            return False
        
        # Stage 2: Baseline Model
        if not self.run_stage(
            "Baseline Model",
            2,
            "exercise_bilstm_stage2_baseline",
            {"epochs": 50, "batch_size": 64, "lr": 0.001}
        ):
            print("\n⚠️ Stage 2 failed. Partial results available.")
            return False
        
        # Stage 3: Optimized Model
        if not self.run_stage(
            "Optimized Model (100 epochs, 0.0005 LR)",
            3,
            "exercise_bilstm_stage3_optimized",
            {"epochs": 100, "batch_size": 64, "lr": 0.0005}
        ):
            print("\n⚠️ Stage 3 failed. Partial results available.")
            return False
        
        # Stage 4: High-LR Exploration (optional fast convergence test)
        if not self.run_stage(
            "Fast Convergence Test (50 epochs, 0.01 LR)",
            4,
            "exercise_bilstm_stage4_fast_lr",
            {"epochs": 50, "batch_size": 64, "lr": 0.01}
        ):
            print("\n⚠️ Stage 4 failed. Check for divergence.")
        
        # Compare results
        best_model = self.compare_models()
        
        # Final summary
        total_duration = (datetime.now() - self.start_time).total_seconds() / 60
        
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETE")
        print("="*80)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Duration: {total_duration:.1f} minutes ({total_duration/60:.1f} hours)")
        print(f"\nResults saved to:")
        for model_dir in self.results.keys():
            print(f"  • results/{model_dir}/")
        
        if best_model:
            print(f"\n🎯 Recommended Production Model: {best_model}")
            print(f"   Use checkpoint: results/{best_model}/exercise_bilstm_best.pt")
        
        return True


def main():
    """Main entry point."""
    pipeline = TrainingPipeline()
    
    try:
        success = pipeline.run_all_stages()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
