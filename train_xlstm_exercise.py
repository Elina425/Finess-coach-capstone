#!/usr/bin/env python3
"""
Train xLSTM Exercise Classifier

This script trains the advanced xLSTM model with:
- Chebyshev interpolation for sequences
- Bidirectional xLSTM encoder
- Multi-task learning (classification + quality)
- Gemma feedback integration

Usage:
    python train_xlstm_exercise.py \
        --data-csv results/riccio_index.csv \
        --feature-dir results/riccio_features \
        --epochs 100 \
        --batch-size 64 \
        --lr 0.0005 \
        --output-dir results/xlstm_model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import argparse
import json
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from collections import defaultdict

# Import our custom modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from fitness_coach.datasets.advanced_video_dataset import VideoExerciseDataset
from fitness_coach.models.xlstm_model import xLSTMExerciseClassifier
from fitness_coach.preprocessing.interpolation import MotionSequenceInterpolator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class xLSTMTrainer:
    """Trainer class for xLSTM exercise model."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def setup_data(self):
        """Load and prepare datasets."""
        logger.info("Loading dataset...")
        
        # Load full dataset
        dataset = VideoExerciseDataset(
            data_source=self.args.data_csv,
            feature_dir=self.args.feature_dir,
            feature_type=self.args.feature_type,
            target_frames=self.args.target_frames,
            interpolation=self.args.interpolation,
            preload_features=self.args.preload_features
        )
        
        logger.info(f"Dataset size: {len(dataset)}")
        logger.info(f"Class distribution: {dataset.get_class_distribution()}")
        
        # Split into train/val/test
        train_size = int(0.6 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        logger.info(f"Split: train={train_size}, val={val_size}, test={test_size}")
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        # Get class weights
        self.class_weights = dataset.get_class_weights(device=self.device)
        logger.info(f"Class weights: {self.class_weights}")
        
        self.class_to_idx = dataset.class_to_idx
        self.idx_to_class = dataset.idx_to_class
    
    def setup_model(self):
        """Initialize model."""
        logger.info("Initializing xLSTM model...")
        
        self.model = xLSTMExerciseClassifier(
            input_size=self.args.input_size,
            hidden_size=self.args.hidden_size,
            num_layers=self.args.num_layers,
            num_classes=len(self.class_to_idx),
            dropout=self.args.dropout,
            bidirectional=True
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def setup_optimizer(self):
        """Initialize optimizer and scheduler."""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        logger.info(f"Optimizer: Adam (lr={self.args.lr})")
        logger.info(f"Scheduler: ReduceLROnPlateau")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device)
            quality = batch['quality'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            class_logits, quality_scores = self.model(features)
            
            # Compute loss
            loss = self.model.get_loss(
                class_logits, quality_scores, labels, quality,
                class_weight=self.args.class_weight,
                quality_weight=self.args.quality_weight
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            preds = class_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if (batch_idx + 1) % self.args.log_interval == 0:
                accuracy = 100 * correct / total
                avg_loss = total_loss / (batch_idx + 1)
                logger.info(
                    f"Epoch [{self.current_epoch + 1}/{self.args.epochs}] "
                    f"Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%"
                )
        
        return total_loss / len(self.train_loader), correct / total
    
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        predictions = defaultdict(list)
        
        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                quality = batch['quality'].to(self.device)
                
                class_logits, quality_scores = self.model(features)
                
                loss = self.model.get_loss(
                    class_logits, quality_scores, labels, quality,
                    class_weight=self.args.class_weight,
                    quality_weight=self.args.quality_weight
                )
                
                total_loss += loss.item()
                preds = class_logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                # Store for metrics
                predictions['preds'].extend(preds.cpu().numpy())
                predictions['labels'].extend(labels.cpu().numpy())
                predictions['probs'].extend(torch.softmax(class_logits, dim=1).cpu().numpy())
                predictions['quality'].extend(quality_scores.squeeze().cpu().numpy())
        
        accuracy = correct / total
        
        return total_loss / len(self.val_loader), accuracy, predictions
    
    def test(self):
        """Test on test set."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        all_quality = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                class_logits, quality_scores = self.model(features)
                
                preds = class_logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(torch.softmax(class_logits, dim=1).cpu().numpy())
                all_quality.extend(quality_scores.squeeze().cpu().numpy())
        
        # Compute metrics
        test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        
        # Per-class metrics
        per_class_metrics = {}
        for class_idx in range(len(self.class_to_idx)):
            mask = np.array(all_labels) == class_idx
            if mask.sum() > 0:
                class_acc = np.mean(np.array(all_preds)[mask] == all_labels[mask])
                per_class_metrics[self.idx_to_class[class_idx]] = {
                    'accuracy': float(class_acc),
                    'count': int(mask.sum())
                }
        
        return {
            'test_accuracy': float(test_acc),
            'per_class_metrics': per_class_metrics,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'quality_scores': all_quality
        }
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        
        best_val_acc = 0
        best_model_path = self.output_dir / 'xlstm_best.pt'
        
        training_history = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(self.args.epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_preds = self.validate()
            
            # Record history
            training_history['epochs'].append(epoch + 1)
            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc)
            
            logger.info(
                f"Epoch {epoch + 1:3d}/{self.args.epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
            )
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), best_model_path)
                logger.info(f"✓ Saved best model (val_acc={val_acc:.4f})")
            
            # Update learning rate
            self.scheduler.step(val_acc)
        
        # Save training history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")
        
        # Test on best model
        logger.info("Testing on best model...")
        self.model.load_state_dict(torch.load(best_model_path))
        test_results = self.test()
        
        # Save test results
        results_path = self.output_dir / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'test_accuracy': test_results['test_accuracy'],
                'per_class_metrics': test_results['per_class_metrics'],
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Test Accuracy: {test_results['test_accuracy']:.4f}")
        logger.info(f"Saved test results to {results_path}")
        logger.info("✓ Training complete!")
        
        return test_results


def main():
    parser = argparse.ArgumentParser(description='Train xLSTM Exercise Classifier')
    
    # Data arguments
    parser.add_argument('--data-csv', type=str, default='results/riccio_index.csv',
                       help='Path to metadata CSV')
    parser.add_argument('--feature-dir', type=str, default='results/riccio_features',
                       help='Directory with precomputed features')
    parser.add_argument('--feature-type', choices=['pose', 'hybrid'], default='pose',
                       help='Feature type to use')
    parser.add_argument('--target-frames', type=int, default=60,
                       help='Target sequence length')
    parser.add_argument('--interpolation', choices=['linear', 'chebyshev', 'spline'],
                       default='chebyshev', help='Interpolation method')
    parser.add_argument('--preload-features', action='store_true',
                       help='Preload all features into memory')
    
    # Model arguments
    parser.add_argument('--input-size', type=int, default=13,
                       help='Input feature dimension')
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='Hidden state size')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of xLSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='L2 regularization')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                       help='Gradient clipping norm (0 = no clipping)')
    parser.add_argument('--class-weight', type=float, default=1.0,
                       help='Weight for classification loss')
    parser.add_argument('--quality-weight', type=float, default=0.5,
                       help='Weight for quality loss')
    
    # Utilities
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of data loading workers')
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Logging interval (batches)')
    parser.add_argument('--output-dir', type=str, default='results/xlstm_model',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run training
    trainer = xLSTMTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
