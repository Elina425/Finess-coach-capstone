"""
Advanced Video Exercise Dataset

Handles:
- Video paths with exercise labels and quality scores
- Metadata (view type, subject ID, etc.)
- Feature types: pose-only, hybrid (pose + visual)
- Interpolation methods: linear, Chebyshev, spline
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import json
import csv


class VideoExerciseDataset(Dataset):
    """
    Advanced dataset for exercise video sequences.
    
    Features:
    - Loads video metadata (paths, labels, quality scores)
    - Handles pose-only and hybrid features
    - Supports multiple interpolation strategies
    - Nyquist-Shannon aware sampling
    """
    
    def __init__(self, 
                 data_source,  # CSV or JSON with metadata
                 feature_dir=None,  # Directory with precomputed features
                 feature_type='pose',  # 'pose' or 'hybrid'
                 target_frames=60,  # Target sequence length
                 interpolation='linear',  # 'linear', 'chebyshev', 'spline'
                 preload_features=False,  # Load all features into memory
                 class_to_idx=None):
        """
        Args:
            data_source: Path to CSV/JSON with metadata
            feature_dir: Directory containing precomputed NPZ features
            feature_type: 'pose' for angles only, 'hybrid' for pose + DINOv3
            target_frames: Resample sequences to this length
            interpolation: 'linear', 'chebyshev', 'spline'
            preload_features: If True, load all features into memory
            class_to_idx: Optional dict mapping class names to indices
        """
        
        self.data_source = Path(data_source)
        self.feature_dir = Path(feature_dir) if feature_dir else None
        self.feature_type = feature_type
        self.target_frames = target_frames
        self.interpolation = interpolation
        self.preload_features = preload_features
        
        # Load metadata
        self.metadata = self._load_metadata(self.data_source)
        
        # Build class mapping
        if class_to_idx is None:
            unique_classes = sorted(set(item.get('label') or item.get('exercise_class') 
                                        for item in self.metadata))
            self.class_to_idx = {c: i for i, c in enumerate(unique_classes)}
        else:
            self.class_to_idx = class_to_idx
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Optional: preload all features
        self.cached_features = {}
        if self.preload_features and self.feature_dir:
            self._preload_all_features()
        
        print(f"✓ Loaded {len(self.metadata)} samples")
        print(f"✓ Classes: {list(self.class_to_idx.keys())}")
        print(f"✓ Target frames: {self.target_frames}")
        print(f"✓ Interpolation: {self.interpolation}")
    
    def _load_metadata(self, path):
        """Load metadata from CSV or JSON"""
        if path.suffix == '.csv':
            return self._load_csv(path)
        elif path.suffix == '.json':
            return self._load_json(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
    
    def _load_csv(self, path):
        """Load from CSV"""
        metadata = []
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle different column names
                item = {
                    'video_path': row.get('video_path') or row.get('path'),
                    'label': row.get('label') or row.get('exercise_class') or row.get('exercise'),
                    'quality': float(row.get('quality', 3.0)),
                    'view_type': row.get('view_type', 'unknown'),
                    'subject_id': row.get('subject_id', row.get('subject', 'unknown')),
                    'split': row.get('split', 'train'),
                }
                metadata.append(item)
        return metadata
    
    def _load_json(self, path):
        """Load from JSON"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _preload_all_features(self):
        """Load all features into memory"""
        print("Preloading all features...")
        for i, item in enumerate(self.metadata):
            if i % max(1, len(self.metadata) // 10) == 0:
                print(f"  Loading {i}/{len(self.metadata)}...")
            features = self._load_features(i)
            if features is not None:
                self.cached_features[i] = features
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """Get a sample"""
        item = self.metadata[idx]
        
        # Load features
        if idx in self.cached_features:
            features = self.cached_features[idx]
        else:
            features = self._load_features(idx)
        
        if features is None:
            # Return zeros if feature loading fails
            features = np.zeros((self.target_frames, 13))
        
        # Ensure correct shape
        features = self._resample_features(features)
        features = self._interpolate(features)
        
        # Load label and quality
        label = self.class_to_idx.get(item['label'], 0)
        quality = float(item.get('quality', 3.0))
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'quality': torch.tensor(quality, dtype=torch.float32),
            'metadata': {
                'video_path': item.get('video_path', 'unknown'),
                'view_type': item.get('view_type', 'unknown'),
                'subject_id': item.get('subject_id', 'unknown'),
                'split': item.get('split', 'unknown'),
            }
        }
    
    def _load_features(self, idx):
        """Load precomputed features from disk"""
        if self.feature_dir is None:
            return None
        
        item = self.metadata[idx]
        video_path = item.get('video_path', '')
        
        # Try different naming conventions
        stem = Path(video_path).stem
        
        # Try .npz file
        if self.feature_type == 'hybrid':
            npz_file = self.feature_dir / f"{stem}_hybrid.npz"
        else:
            npz_file = self.feature_dir / f"{stem}_pose.npz"
        
        if npz_file.exists():
            try:
                data = np.load(npz_file)
                if 'features' in data:
                    return data['features']
                elif 'X' in data:
                    return data['X']
                else:
                    # Try to get first array
                    return next(iter(data.values()))
            except Exception as e:
                print(f"Warning: Could not load {npz_file}: {e}")
        
        return None
    
    def _resample_features(self, features):
        """
        Resample feature sequence to target length.
        
        Uses linear interpolation. This is Nyquist-Shannon aware:
        - If original is sparse, we preserve information
        - If original is dense, we downsample uniformly
        """
        current_len = features.shape[0]
        
        if current_len == self.target_frames:
            return features
        
        # Generate indices for resampling
        indices = np.linspace(0, current_len - 1, self.target_frames)
        resampled = np.zeros((self.target_frames, features.shape[1]))
        
        # Linear interpolation
        for i, idx in enumerate(indices):
            idx_low = int(np.floor(idx))
            idx_high = int(np.ceil(idx))
            idx_high = min(idx_high, current_len - 1)
            
            weight = idx - idx_low
            
            if idx_low == idx_high:
                resampled[i] = features[idx_low]
            else:
                resampled[i] = (1 - weight) * features[idx_low] + weight * features[idx_high]
        
        return resampled
    
    def _interpolate(self, features):
        """Apply additional interpolation if specified"""
        if self.interpolation == 'chebyshev':
            return self._chebyshev_interpolate(features)
        elif self.interpolation == 'spline':
            return self._spline_interpolate(features)
        else:
            # Linear was already done in resampling
            return features
    
    def _chebyshev_interpolate(self, features):
        """
        Chebyshev polynomial interpolation.
        
        Benefits:
        - Minimizes Runge oscillation at edges
        - Better numerical stability than equidistant points
        - Justified for bounded motion sequences
        """
        try:
            current_len = features.shape[0]
            
            # Generate Chebyshev nodes in [-1, 1]
            k = np.arange(1, current_len + 1)
            chebyshev_nodes = np.cos((2 * k - 1) * np.pi / (2 * current_len))
            
            # Map to [0, current_len - 1]
            x_old = (chebyshev_nodes + 1) / 2 * (current_len - 1)
            x_new = np.linspace(0, current_len - 1, self.target_frames)
            
            interpolated = np.zeros((self.target_frames, features.shape[1]))
            
            # Interpolate each dimension
            for dim in range(features.shape[1]):
                # Fit polynomial (use lower degree to avoid overfitting)
                deg = min(current_len - 1, 10)
                coeff = np.polyfit(x_old, features[:, dim], deg)
                poly = np.poly1d(coeff)
                interpolated[:, dim] = poly(x_new)
            
            return interpolated
        except Exception:
            # Fallback to original if interpolation fails
            return features
    
    def _spline_interpolate(self, features):
        """
        Spline interpolation for smooth motion curves.
        
        Good for biomechanical data where smooth motion is expected.
        """
        try:
            from scipy.interpolate import UnivariateSpline
            
            current_len = features.shape[0]
            x_old = np.linspace(0, current_len - 1, current_len)
            x_new = np.linspace(0, current_len - 1, self.target_frames)
            
            interpolated = np.zeros((self.target_frames, features.shape[1]))
            
            for dim in range(features.shape[1]):
                # Use smoothing spline (s=None means interpolating spline)
                spline = UnivariateSpline(x_old, features[:, dim], s=None)
                interpolated[:, dim] = spline(x_new)
            
            return interpolated
        except Exception:
            # Fallback if spline interpolation fails
            return features
    
    def get_class_distribution(self):
        """Get distribution of classes in dataset"""
        dist = {}
        for item in self.metadata:
            label = item.get('label') or item.get('exercise_class')
            dist[label] = dist.get(label, 0) + 1
        return dist
    
    def get_class_weights(self, device='cpu'):
        """Get class weights for imbalanced dataset (inverse frequency)"""
        dist = self.get_class_distribution()
        total = len(self.metadata)
        
        weights = []
        for i in range(len(self.class_to_idx)):
            class_name = self.idx_to_class[i]
            count = dist.get(class_name, 0)
            weight = total / (len(self.class_to_idx) * (count + 1))
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32, device=device)
