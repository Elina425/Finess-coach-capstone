# Advanced Fitness Coach Training Pipeline
## Implementation Plan: xLSTM + Pretrained Vision + Gemma-X

---

## Phase 1: data Loading & Preprocessing

### 1.1 Enhanced Dataset Classes

**File:** `fitness_coach/datasets/advanced_video_dataset.py`

```python
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import json

class VideoExerciseDataset(Dataset):
    """
    Advanced dataset that handles:
    - Video paths with exercise labels
    - Quality scores
    - Metadata (view type, subject ID)
    """
    
    def __init__(self, 
                 data_source,  # CSV or JSON with video_path, label, quality, etc.
                 video_dir=None,
                 feature_type='hybrid',  # 'pose' or 'hybrid' (pose + visual)
                 target_frames=60,
                 interpolation='linear',
                 transform=None):
        
        self.data_source = data_source
        self.video_dir = video_dir
        self.feature_type = feature_type
        self.target_frames = target_frames
        self.interpolation = interpolation
        self.transform = transform
        
        # Load metadata
        if isinstance(data_source, str):
            self.metadata = self._load_metadata(data_source)
        else:
            self.metadata = data_source
    
    def _load_metadata(self, path):
        """Load from CSV or JSON"""
        if path.endswith('.csv'):
            import pandas as pd
            return pd.read_csv(path).to_dict('records')
        else:
            with open(path) as f:
                return json.load(f)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Extract components
        video_path = item.get('video_path')
        label = item.get('label') or item.get('exercise_class')
        quality = item.get('quality', 0)
        
        # Load features (pose or hybrid)
        if self.feature_type == 'pose':
            features = self._load_pose_features(video_path)
        else:  # hybrid
            features = self._load_hybrid_features(video_path)
        
        # Resample to target length
        features = self._resample_features(features)
        
        # Apply interpolation if needed
        features = self._interpolate(features)
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'quality': torch.tensor(quality, dtype=torch.float32),
            'metadata': {
                'video_path': video_path,
                'view_type': item.get('view_type', 'unknown'),
                'subject_id': item.get('subject_id', 'unknown')
            }
        }
    
    def _load_pose_features(self, video_path):
        """Load pose features (joint angles, coordinates)"""
        # Load from precomputed NPZ or extract on-the-fly
        pass
    
    def _load_hybrid_features(self, video_path):
        """Load pose + DINOv3 visual embeddings"""
        pose = self._load_pose_features(video_path)
        visual = self._load_visual_embeddings(video_path)
        return np.concatenate([pose, visual], axis=-1)
    
    def _load_visual_embeddings(self, video_path):
        """Extract DINOv3 embeddings"""
        pass
    
    def _resample_features(self, features):
        """Resample sequence to target length (Nyquist-Shannon aware)"""
        current_len = features.shape[0]
        if current_len == self.target_frames:
            return features
        
        indices = np.linspace(0, current_len - 1, self.target_frames)
        resampled = np.zeros((self.target_frames, features.shape[1]))
        
        for i, idx in enumerate(indices):
            idx_low = int(np.floor(idx))
            idx_high = int(np.ceil(idx))
            weight = idx - idx_low
            
            if idx_low == idx_high:
                resampled[i] = features[idx_low]
            else:
                resampled[i] = (1 - weight) * features[idx_low] + weight * features[idx_high]
        
        return resampled
    
    def _interpolate(self, features):
        """Apply Chebyshev or spline interpolation for smoother motion"""
        if self.interpolation == 'chebyshev':
            return self._chebyshev_interpolate(features)
        elif self.interpolation == 'spline':
            return self._spline_interpolate(features)
        else:
            return features  # Already linear interpolated
    
    def _chebyshev_interpolate(self, features):
        """Chebyshev polynomial interpolation for smooth motion"""
        from scipy.interpolate import ChebyshevGrid
        # Implementation details...
        return features
    
    def _spline_interpolate(self, features):
        """Spline interpolation for smooth motion curves"""
        from scipy.interpolate import UnivariateSpline
        # Implementation details...
        return features
```

---

## Phase 2: Frame Sampling & Feature Extraction

### 2.1 Frame Sampler

**File:** `fitness_coach/preprocessing/frame_sampler.py`

```python
class FrameSampler:
    """
    Handles proper frame sampling with Nyquist-Shannon considerations.
    
    Theory:
    - Nyquist frequency = 0.5 * frame_rate
    - If motion changes faster than 2x sampled rate, you lose info
    - Solution: sample sufficient frames or use motion detection
    """
    
    def __init__(self, 
                 target_frames=60,
                 method='uniform',  # 'uniform', 'motion_adaptive', 'motion_aware'
                 min_fps=15,
                 max_fps=30):
        
        self.target_frames = target_frames
        self.method = method
        self.min_fps = min_fps
        self.max_fps = max_fps
    
    def sample_frames(self, video_path, fps=None):
        """
        Sample frames from video respecting Nyquist constraints.
        
        Returns:
            frames: ndarray of shape (target_frames, H, W, 3)
            metadata: dict with actual_fps, nyquist_freq, satisfies_nyquist
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Nyquist check
        nyquist_freq = actual_fps / 2  # Max frequency we can capture
        sampling_rate = total_frames / self.target_frames
        nyquist_satisfied = actual_fps >= 2 * (1 / sampling_rate)
        
        if self.method == 'uniform':
            frame_indices = np.linspace(0, total_frames - 1, self.target_frames, dtype=int)
        elif self.method == 'motion_adaptive':
            frame_indices = self._motion_adaptive_sampling(cap, total_frames)
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.target_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        
        return np.array(frames), {
            'actual_fps': actual_fps,
            'nyquist_freq': nyquist_freq,
            'satisfies_nyquist': nyquist_satisfied,
            'frame_indices': frame_indices
        }
    
    def _motion_adaptive_sampling(self, cap, total_frames):
        """Sample more frames where motion is detected"""
        # Read all frames, compute optical flow
        # Sample more densely in high-motion regions
        pass
```

### 2.2 Feature Extractor

**File:** `fitness_coach/preprocessing/feature_extractor.py`

```python
class PoseFeatureExtractor:
    """Extract pose-based features (joints, angles, distances)"""
    
    def __init__(self, keypoint_detector='mediapipe'):
        self.detector = keypoint_detector
    
    def extract(self, frames):
        """
        Extract pose landmarks from frames.
        
        Returns:
            features: ndarray of shape (num_frames, num_joints * 3 or custom)
        """
        pass

class HybridFeatureExtractor:
    """Extract pose + visual features using DINOv3"""
    
    def __init__(self,  backbone='dinov3_vitlarge'):
        self.pose_extractor = PoseFeatureExtractor()
        self.visual_encoder = self._load_dinov3(backbone)
    
    def extract(self, frames):
        """Extract both pose and visual features"""
        pose_features = self.pose_extractor.extract(frames)
        visual_features = self._extract_visual_embeddings(frames)
        return np.concatenate([pose_features, visual_features], axis=-1)
    
    def _load_dinov3(self, backbone):
        """Load pretrained DINOv3 model"""
        # Use timm or official DINOv3 checkpoints
        pass
    
    def _extract_visual_embeddings(self, frames):
        """Extract DINOv3 embeddings for each frame"""
        # Batch process frames through DINOv3
        pass
```

---

## Phase 3: Interpolation & Resampling

### 3.1 Interpolation Utilities

**File:** `fitness_coach/preprocessing/interpolation.py`

```python
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline, interp1d

class MotionSequenceInterpolator:
    """
    Properly interpolate motion sequences using different methods.
    
    Theory:
    - Linear: fast, okay for smooth motion
    - Chebyshev: uses Chebyshev polynomials, avoids Runge oscillation
    - Spline: smooth curves, good for biomechanical data
    """
    
    @staticmethod
    def linear_interpolate(sequence, target_length):
        """
        Linear interpolation to target length.
        
        Baseline method, simple and fast.
        """
        current_len = sequence.shape[0]
        if current_len == target_length:
            return sequence
        
        indices = np.linspace(0, current_len - 1, target_length)
        interpolated = np.zeros((target_length, sequence.shape[1]))
        
        for i, idx in enumerate(indices):
            idx_low = int(np.floor(idx))
            idx_high = min(int(np.ceil(idx)), current_len - 1)
            weight = idx - idx_low
            
            interpolated[i] = (1 - weight) * sequence[idx_low] + weight * sequence[idx_high]
        
        return interpolated
    
    @staticmethod
    def chebyshev_interpolate(sequence, target_length):
        """
        Chebyshev polynomial interpolation.
        
        Justification:
        - Minimizes Runge oscillation (overfitting at edges)
        - Better for bounded intervals
        - Uses Chebyshev nodes instead of uniformly spaced points
        """
        current_len = sequence.shape[0]
        
        # Generate Chebyshev nodes in [0, 1]
        k = np.arange(1, current_len + 1)
        chebyshev_nodes = np.cos((2 * k - 1) * np.pi / (2 * current_len))
        
        # Map to [0, current_len - 1]
        x_old = (chebyshev_nodes + 1) / 2 * (current_len - 1)
        x_new = np.linspace(0, current_len - 1, target_length)
        
        interpolated = np.zeros((target_length, sequence.shape[1]))
        
        for dim in range(sequence.shape[1]):
            # Fit Chebyshev polynomial
            coeff = np.polyfit(x_old, sequence[:, dim], min(current_len - 1, 10))
            poly = np.poly1d(coeff)
            interpolated[:, dim] = poly(x_new)
        
        return interpolated
    
    @staticmethod
    def spline_interpolate(sequence, target_length, smooth=None):
        """
        Spline interpolation for smooth motion curves.
        
        Good for biomechanical data where smooth motion is expected.
        """
        current_len = sequence.shape[0]
        x_old = np.linspace(0, current_len - 1, current_len)
        x_new = np.linspace(0, current_len - 1, target_length)
        
        interpolated = np.zeros((target_length, sequence.shape[1]))
        
        for dim in range(sequence.shape[1]):
            # Use smoothing spline
            if smooth is None:
                smooth = current_len  # Default: interpolating spline
            
            spline = UnivariateSpline(x_old, sequence[:, dim], s=smooth)
            interpolated[:, dim] = spline(x_new)
        
        return interpolated

# Usage
# interpolator = MotionSequenceInterpolator()
# resampled = interpolator.chebyshev_interpolate(sequence, target_length=60)
```

---

## Phase 4: xLSTM Temporal Model

### 4.1 xLSTM Implementation

**File:** `fitness_coach/models/xlstm_model.py`

```python
import torch
import torch.nn as nn

class xLSTMCell(nn.Module):
    """
    Extended LSTM cell with enhanced gating and state management.
    
    Compared to standard LSTM:
    - Richer gating mechanisms
    - Optional exponential stabilization
    - Designed for longer sequences
    """
    
    def __init__(self, input_size, hidden_size, bias=True, 
                 use_exp_gate=True, use_norm=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_exp_gate = use_exp_gate
        self.use_norm = use_norm
        
        # Standard LSTM weights
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        
        if bias:
            self.bias_ih = nn.Parameter(torch.zeros(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.zeros(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        
        # Optional exponential gate (helps with vanishing gradients)
        if use_exp_gate:
            self.exp_gate_scale = nn.Parameter(torch.ones(1))
        
        # Optional layer norm
        if use_norm:
            self.norm = nn.LayerNorm(hidden_size)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.orthogonal_(p)
            else:
                nn.init.zeros_(p)
    
    def forward(self, x, h_prev, c_prev):
        """
        Forward pass of xLSTM cell.
        
        Args:
            x: input (batch, input_size)
            h_prev: hidden state (batch, hidden_size)
            c_prev: cell state (batch, hidden_size)
        
        Returns:
            h_new, c_new
        """
        # Compute gates
        gates = torch.mm(x, self.weight_ih.t()) + torch.mm(h_prev, self.weight_hh.t())
        
        if self.bias_ih is not None:
            gates = gates + self.bias_ih
        
        # Split into i, f, g, o gates
        i, f, g, o = gates.chunk(4, 1)
        
        # xLSTM: exponential stabilization
        if self.use_exp_gate:
            i = torch.exp(i * self.exp_gate_scale)
            f = torch.exp(f * self.exp_gate_scale)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        # Cell state
        c_new = f * c_prev + i * g
        
        # Hidden state
        h_new = o * torch.tanh(c_new)
        
        # Optional normalization
        if self.use_norm:
            h_new = self.norm(h_new)
        
        return h_new, c_new


class xLSTM(nn.Module):
    """
    Multi-layer extended LSTM for temporal sequence modeling.
    
    Architecture:
        Input (batch, seq_len, input_size)
            ↓
        [xLSTM Layer 1] → hidden representation
            ↓
        [xLSTM Layer 2] → higher level representation (optional)
            ↓
        Output (batch, seq_len, hidden_size) or (batch, hidden_size)
    """
    
    def __init__(self, input_size, hidden_size, num_layers=2, 
                 dropout=0.3, bidirectional=False,
                 use_exp_gate=True, use_norm=True):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout if num_layers > 1 else 0
        self.bidirectional = bidirectional
        
        # Create xLSTM layers
        self.layers = nn.ModuleList()
        
        for layer in range(num_layers):
            input_sz = input_size if layer == 0 else hidden_size
            
            # Forward direction
            self.layers.append(
                xLSTMCell(input_sz, hidden_size,
                         use_exp_gate=use_exp_gate,
                         use_norm=use_norm)
            )
            
            # Backward direction (if bidirectional)
            if bidirectional:
                self.layers.append(
                    xLSTMCell(input_sz, hidden_size,
                             use_exp_gate=use_exp_gate,
                             use_norm=use_norm)
                )
            
            if layer < num_layers - 1:
                self.dropout_layer = nn.Dropout(self.dropout) if self.dropout > 0 else None
    
    def forward(self, x, states=None):
        """
        Forward pass through xLSTM stack.
        
        Args:
            x: (batch, seq_len, input_size)
            states: optional initial hidden states
        
        Returns:
            outputs: (batch, seq_len, hidden_size or 2*hidden_size if bidirectional)
            (h_final, c_final): final hidden and cell states
        """
        batch_size, seq_len, _ = x.size()
        
        # Initialize states
        if states is None:
            device = x.device
            h_states = [torch.zeros(batch_size, self.hidden_size, device=device) 
                       for _ in range(self.num_layers * (2 if self.bidirectional else 1))]
            c_states = [torch.zeros(batch_size, self.hidden_size, device=device) 
                       for _ in range(self.num_layers * (2 if self.bidirectional else 1))]
        else:
            h_states, c_states = states
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_size)
            layer_input = x_t
            
            for layer in range(self.num_layers):
                # Forward pass
                cell_idx = layer * (2 if self.bidirectional else 1)
                h_states[cell_idx], c_states[cell_idx] = self.layers[cell_idx](
                    layer_input, h_states[cell_idx], c_states[cell_idx]
                )
                layer_input = h_states[cell_idx]
                
                # Backward pass (if bidirectional)
                if self.bidirectional:
                    cell_idx_bwd = layer * 2 + 1
                    h_states[cell_idx_bwd], c_states[cell_idx_bwd] = self.layers[cell_idx_bwd](
                        layer_input, h_states[cell_idx_bwd], c_states[cell_idx_bwd]
                    )
                    layer_input = torch.cat([h_states[cell_idx], h_states[cell_idx_bwd]], dim=1)
                
                # Dropout
                if layer < self.num_layers - 1 and self.dropout_layer is not None:
                    layer_input = self.dropout_layer(layer_input)
            
            outputs.append(layer_input.unsqueeze(1))
        
        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden_size or 2*hidden_size)
        
        return outputs, (h_states, c_states)


class xLSTMExerciseClassifier(nn.Module):
    """
    Complete model: xLSTM + classification + quality prediction head.
    
    Architecture:
        Input features (batch, seq_len, feature_dim)
            ↓
        xLSTM Encoder (bidirectional, 2 layers)
            ↓
        ┌─────────────────────────┬─────────────────────────┐
        ↓                         ↓
    Exercise Head            Quality Head
    (5 classes)              (regression 0-5)
        ↓                         ↓
    Classification logits   Quality score
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 num_classes=5, dropout=0.3):
        super().__init__()
        
        # xLSTM encoder
        self.xlstm = xLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            use_exp_gate=True,
            use_norm=True
        )
        
        xlstm_output_size = hidden_size * 2 if True else hidden_size
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.class_head = nn.Sequential(
            nn.Linear(xlstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Quality regression head
        self.quality_head = nn.Sequential(
            nn.Linear(xlstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Output in [0, 1], then scale to [0, 5]
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, input_size)
        
        Returns:
            class_logits: (batch, num_classes)
            quality_scores: (batch, 1) in [0, 5]
        """
        # xLSTM encoding
        xlstm_out, _ = self.xlstm(x)  # (batch, seq_len, 2*hidden_size)
        
        # Global average pooling
        pooled = self.pool(xlstm_out.transpose(1, 2)).squeeze(-1)  # (batch, 2*hidden_size)
        
        # Classification head
        class_logits = self.class_head(pooled)
        
        # Quality head
        quality_raw = self.quality_head(pooled)
        quality_scores = quality_raw * 5.0  # Scale to [0, 5]
        
        return class_logits, quality_scores
```

---

## Phase 5: Training Loop

### 5.1 Training Script

**File:** `train_xlstm_exercise.py`

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from pathlib import Path
import argparse

def train_xlstm_model(args):
    """
    Training loop for xLSTM exercise classifier.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    # train_dataset = VideoExerciseDataset(...)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model
    from fitness_coach.models.xlstm_model import xLSTMExerciseClassifier
    
    model = xLSTMExerciseClassifier(
        input_size=args.input_size,  # 13 (pose) or 13+384 (hybrid with DINOv3)
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=5,
        dropout=args.dropout
    ).to(device)
    
    # Loss functions
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Training loop
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            quality = batch['quality'].to(device)
            
            # Forward pass
            class_logits, quality_scores = model(features)
            
            # Compute loss
            loss_class = ce_loss(class_logits, labels)
            loss_quality = mse_loss(quality_scores.squeeze(), quality)
            loss = loss_class + 0.5 * loss_quality
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_acc = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                labels = batch['label'].to(device)
                
                class_logits, _ = model(features)
                preds = class_logits.argmax(dim=1)
                val_acc += (preds == labels).sum().item()
        
        val_acc /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1:3d}  loss={train_loss/len(train_loader):.4f}  val_acc={val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output_dir / 'xlstm_best.pt')
    
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--input-size', type=int, default=13)  # pose-only
    parser.add_argument('--output-dir', type=Path, default=Path('results/xlstm_exercise'))
    parser.add_argument('--data-source', type=str, required=True)
    parser.add_argument('--feature-type', choices=['pose', 'hybrid'], default='pose')
    parser.add_argument('--interpolation',  choices=['linear', 'chebyshev', 'spline'], default='chebyshev')
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    train_xlstm_model(args)
```

---

## Phase 6: Feedback Generation with Gemma-X

### 6.1 Feedback Generator

**File:** `fitness_coach/inference/gemma_feedback.py`

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class GemmaFeedbackGenerator:
    """
    Use Gemma-X to generate natural language feedback.
    
    Input: predicted exercise, quality score, problematic angles
    Output: Natural language feedback ("Your squat depth is shallow...")
    """
    
    def __init__(self, model_name="google/gemma-2b-it", device='cpu'):
        """
        Initialize Gemma model.
        
        Available models:
        - google/gemma-2b-it (lightweight, ideal for edge)
        - google/gemma-7b-it (larger, better quality)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map=device
        )
        self.device = device
    
    def generate_feedback(self, exercise_class, quality_score, 
                         problematic_joints=None, biomechanics_dict=None):
        """
        Generate feedback for exercise performance.
        
        Args:
            exercise_class: str, e.g. "squat"
            quality_score: float [0, 5]
            problematic_joints: list of joint names with high error
            biomechanics_dict: dict with angle measurements
        
        Returns:
            feedback: str, natural language feedback
        """
        
        # Build prompt
        prompt = self._build_prompt(
            exercise_class, quality_score, problematic_joints, biomechanics_dict
        )
        
        # Generate with Gemma
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            early_stopping=True
        )
        
        feedback = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        feedback = feedback.replace(prompt, "").strip()  # Remove prompt from output
        
        return feedback
    
    def _build_prompt(self, exercise, quality, joints, biomechanics):
        """Build prompt for Gemma"""
        
        prompt = f"""You are a fitness coach analyzing exercise performance.

Exercise: {exercise}
Quality score: {quality:.1f}/5.0
"""
        
        if quality  < 2.5:
            prompt += "Form quality: POOR. Significant improvements needed.\n"
        elif quality < 3.5:
            prompt += "Form quality: FAIR. Some adjustments recommended.\n"
        else:
            prompt += "Form quality: GOOD. Minor refinements possible.\n"
        
        if joints:
            prompt += f"Problematic joints: {', '.join(joints)}\n"
        
        if biomechanics:
            prompt += "Key measurements:\n"
            for joint, angle in list(biomechanics.items())[:3]:
                prompt += f"  - {joint}: {angle}°\n"
        
        prompt += f"\nProvide 1-2 specific, actionable coaching tips to improve this {exercise}:\n"
        
        return prompt

# Usage
# generator = GemmaFeedbackGenerator()
# feedback = generator.generate_feedback(
#     "squat", 3.2, 
#     problematic_joints=['hip', 'knee'],
#     biomechanics_dict={'hip_angle': 85, 'knee_angle': 92, 'ankle_angle': 78}
# )
```

---

## Phase 7: Complete Inference Pipeline

### 7.1 End-to-End Inference

**File:** `train_complete_inference_pipeline.py`

```python
import torch
from pathlib import Path
import json
import cv2
import numpy as np

class CompleteExerciseInferencePipeline:
    """
    End-to-end inference: video → features → xLSTM → quality → Gemma feedback
    """
    
    def __init__(self,  
                 xlstm_checkpoint,
                 feature_type='hybrid',
                 device='cpu'):
        
        self.device = device
        self.feature_type = feature_type
        
        # Load xLSTM model
        from fitness_coach.models.xlstm_model import xLSTMExerciseClassifier
        self.model = xLSTMExerciseClassifier(...)
        self.model.load_state_dict(torch.load(xlstm_checkpoint, map_location=device))
        self.model.eval()
        
        # Load Gemma feedback generator
        from fitness_coach.inference.gemma_feedback import GemmaFeedbackGenerator
        self.feedback_gen = GemmaFeedbackGenerator(device=device)
        
        # Feature extractors
        if feature_type == 'hybrid':
            from fitness_coach.preprocessing.feature_extractor import HybridFeatureExtractor
            self.extractor = HybridFeatureExtractor()
        else:
            from fitness_coach.preprocessing.feature_extractor import PoseFeatureExtractor
            self.extractor = PoseFeatureExtractor()
        
        # Exercise classes
        self.classes = ['barbell_biceps_curl', 'hammer_curl', 'push_up', 'shoulder_press', 'squat']
    
    def process_video(self, video_path):
        """
        Process entire video and generate feedback.
        """
        # Step 1: Sample frames
        from fitness_coach.preprocessing.frame_sampler import FrameSampler
        sampler = FrameSampler(target_frames=60, method='uniform')
        frames, metadata = sampler.sample_frames(video_path)
        
        # Step 2: Extract features
        features = self.extractor.extract(frames)
        
        # Step 3: Resample and interpolate
        from fitness_coach.preprocessing.interpolation import MotionSequenceInterpolator
        interpolator = MotionSequenceInterpolator()
        features = interpolator.chebyshev_interpolate(features, target_length=60)
        
        # Step 4: Inference with xLSTM
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            class_logits, quality_scores = self.model(features_tensor)
        
        # Step 5: Extract predictions
        class_pred = class_logits.argmax(dim=1).item()
        quality = quality_scores.squeeze().item()
        confidence = torch.softmax(class_logits, dim=1)[0, class_pred].item()
        
        # Step 6: Generate feedback with Gemma
        exercise_name = self.classes[class_pred]
        feedback = self.feedback_gen.generate_feedback(
            exercise_name, quality,
            problematic_joints=self._identify_problematic_joints(features),
            biomechanics_dict=self._extract_biomechanics(features)
        )
        
        return {
            'exercise': exercise_name,
            'confidence': confidence,
            'quality_score': quality,
            'feedback': feedback,
            'metadata': metadata
        }
    
    def _identify_problematic_joints(self, features):
        """Identify which joints show poor form"""
        # Analyze motion patterns
        pass
    
    def _extract_biomechanics(self, features):
        """Extract key biomechanical measurements"""
        pass
```

---

## Implementation Steps

1. **Create dataset class** → `fitness_coach/datasets/advanced_video_dataset.py`
2. **Implement frame sampler** → `fitness_coach/preprocessing/frame_sampler.py`
3. **Implement interpolation** → `fitness_coach/preprocessing/interpolation.py`
4. **Build xLSTM model** → `fitness_coach/models/xlstm_model.py`
5. **Build training script** → `train_xlstm_exercise.py`
6. **Integrate Gemma** → `fitness_coach/inference/gemma_feedback.py`
7. **Complete inference pipeline** → `inference_xlstm_complete.py`
8. **Run tests & validation**

---

**Next:** Choose which phase you'd like me to implement first!

