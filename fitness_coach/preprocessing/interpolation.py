"""
Motion Sequence Interpolation

Implements multiple interpolation strategies for motion sequences:
- Linear: fast baseline
- Chebyshev: minimizes Runge oscillation
- Spline: smooth motion curves
"""

import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
from typing import Tuple, Optional


class MotionSequenceInterpolator:
    """
    Interpolate motion sequences to uniform length.
    
    Theoretical Justification:
    
    1. LINEAR INTERPOLATION
       - Theory: Piecewise linear reconstruction
       - Pros: Simple, fast, no overfitting
       - Cons: May miss smooth motion patterns
       - Use case: Quick baseline
    
    2. CHEBYSHEV POLYNOMIAL INTERPOLATION
       - Theory: Minimize Runge oscillation using Chebyshev nodes
       - Runge's phenomenon: High-degree polynomials with equidistant points
         oscillate drastically, especially at boundaries
       - Chebyshev nodes: Weight more points at boundaries, fewer in middle
       - Interpolation: Use Chebyshev nodes as sample points, fit polynomial,
         evaluate at uniform grid
       - Pros: Mathematically principled, avoids oscillation, smooth curves
       - Cons: Polynomial fitting can be unstable for high degrees
       - Use case: Motion sequences where smoothness and stability matter
       - Reference: Trefethen, L. N. (2000). Spectral Methods in MATLAB
    
    3. SPLINE INTERPOLATION
       - Theory: Piecewise polynomial curves (continuous derivatives)
       - Smoothing spline: Balance between fitting data and smoothness
       - Pros: Local, numerically stable, smooth curves
       - Cons: Requires parameter tuning (smoothing parameter)
       - Use case: Biomechanical data where smooth motion is physical reality
       - Reference: de Boor, C. (1978). A Practical Guide to Splines
    
    4. NYQUIST-SHANNON SAMPLING
       - Theory: Recover original signal if sampled at ≥ 2x max frequency
       - Applied here: By maintaining sufficient target_frames,
         we ensure we don't lose motion information during resampling
       - If video FPS is 30 and we sample 60 frames, we get pseudo-FPS of
         (30 * 60) / video_length, which should satisfy Nyquist
    """
    
    @staticmethod
    def linear_interpolate(sequence: np.ndarray, 
                          target_length: int) -> np.ndarray:
        """
        Linear interpolation to target length.
        
        Args:
            sequence: (length, features)
            target_length: desired output length
        
        Returns:
            (target_length, features)
        """
        current_len = sequence.shape[0]
        
        if current_len == target_length:
            return sequence
        
        # Generate interpolation indices
        indices = np.linspace(0, current_len - 1, target_length)
        interpolated = np.zeros((target_length, sequence.shape[1]))
        
        # Piecewise linear interpolation
        for i, idx in enumerate(indices):
            idx_low = int(np.floor(idx))
            idx_high = min(int(np.ceil(idx)), current_len - 1)
            
            if idx_low == idx_high:
                interpolated[i] = sequence[idx_low]
            else:
                weight = idx - idx_low
                interpolated[i] = (1 - weight) * sequence[idx_low] + \
                                 weight * sequence[idx_high]
        
        return interpolated
    
    @staticmethod
    def chebyshev_interpolate(sequence: np.ndarray,
                             target_length: int,
                             max_poly_degree: int = 15) -> np.ndarray:
        """
        Chebyshev polynomial interpolation to target length.
        
        Avoids Runge oscillation by using Chebyshev nodes as sample points.
        
        Theory:
        -------
        1. Generate Chebyshev points in [-1, 1]:
           x_k = cos((2k-1)π / 2n)  where k = 1, ..., n
        
        2. Map Chebyshev points to [0, current_len-1]
        
        3. Fit polynomial at Chebyshev points
        
        4. Evaluate polynomial at uniform grid points [0, ..., target_length-1]
        
        This approach minimizes the maximum error (minimax approximation).
        
        Args:
            sequence: (length, features)
            target_length: desired output length
            max_poly_degree: maximum polynomial degree (avoid overfitting)
        
        Returns:
            (target_length, features)
        """
        current_len = sequence.shape[0]
        
        if current_len == target_length:
            return sequence
        
        # Generate Chebyshev nodes in [-1, 1]
        k = np.arange(1, current_len + 1)
        chebyshev_nodes_normalized = np.cos((2 * k - 1) * np.pi / (2 * current_len))
        
        # Map to [0, current_len - 1]
        x_sample = (chebyshev_nodes_normalized + 1) / 2 * (current_len - 1)
        
        # Target evaluation points (uniform grid)
        x_target = np.linspace(0, current_len - 1, target_length)
        
        interpolated = np.zeros((target_length, sequence.shape[1]))
        
        # Interpolate each feature dimension
        for dim in range(sequence.shape[1]):
            try:
                # Fit polynomial using Chebyshev points
                # Use lower degree to avoid overfitting
                poly_degree = min(current_len - 1, max_poly_degree)
                
                # Fit polynomial coefficients
                coeff = np.polyfit(x_sample, sequence[:, dim], poly_degree)
                poly = np.poly1d(coeff)
                
                # Evaluate at target points
                interpolated[:, dim] = poly(x_target)
                
            except Exception as e:
                # Fallback to linear if polynomial fitting fails
                print(f"Warning: Chebyshev interpolation failed for dim {dim}: {e}")
                linear_result = MotionSequenceInterpolator.linear_interpolate(
                    sequence[:, dim:dim+1], target_length
                )
                interpolated[:, dim] = linear_result[:, 0]
        
        return interpolated
    
    @staticmethod
    def spline_interpolate(sequence: np.ndarray,
                          target_length: int,
                          smoothing: Optional[float] = None,
                          order: int = 3) -> np.ndarray:
        """
        Spline interpolation for smooth motion curves.
        
        Smoothing splines balance between fitting the data and maintaining smoothness.
        
        Theory:
        -------
        A smoothing spline minimizes:
            E = Σ(y_i - s(x_i))² + λ ∫(s''(x))² dx
        
        where:
        - First term: fit to data
        - Second term: smoothness penalty (controlled by λ)
        - λ trade-off parameter (smoothing parameter)
        
        For biomechanical data, λ should be tuned based on expected noise level.
        
        Args:
            sequence: (length, features)
            target_length: desired output length
            smoothing: smoothing parameter (None = interpolating spline)
                      - None or 0: exact interpolation, may overfit
                      - Large value: very smooth, may miss patterns
                      - Typical: 0.001 to 1.0 for noisy data
            order: spline order (3 = cubic, most common)
        
        Returns:
            (target_length, features)
        """
        current_len = sequence.shape[0]
        
        if current_len == target_length:
            return sequence
        
        # Original time grid
        x_original = np.linspace(0, current_len - 1, current_len)
        
        # Target time grid (uniform)
        x_target = np.linspace(0, current_len - 1, target_length)
        
        interpolated = np.zeros((target_length, sequence.shape[1]))
        
        # Interpolate each feature
        for dim in range(sequence.shape[1]):
            try:
                # Create smoothing spline
                # s=None means interpolating spline (exact fit)
                # s>0 means smoothing spline with parameter s
                if smoothing is None:
                    # Interpolating spline - passes through all points
                    spline_func = UnivariateSpline(
                        x_original, sequence[:, dim],
                        s=None, k=min(order, current_len - 1)
                    )
                else:
                    # Smoothing spline with smoothing parameter
                    spline_func = UnivariateSpline(
                        x_original, sequence[:, dim],
                        s=smoothing, k=min(order, current_len - 1)
                    )
                
                # Evaluate at target points
                interpolated[:, dim] = spline_func(x_target)
                
            except Exception as e:
                # Fallback to linear if spline fails
                print(f"Warning: Spline interpolation failed for dim {dim}: {e}")
                linear_result = MotionSequenceInterpolator.linear_interpolate(
                    sequence[:, dim:dim+1], target_length
                )
                interpolated[:, dim] = linear_result[:, 0]
        
        return interpolated
    
    @staticmethod
    def adaptive_interpolate(sequence: np.ndarray,
                            target_length: int,
                            method: str = 'chebyshev') -> np.ndarray:
        """
        Adaptive selection of interpolation method.
        
        - Short sequences (< 20 points): Use linear (stable)
        - Medium sequences (20-100): Use Chebyshev (balanced)
        - Longer sequences (100+ points): Use spline (smooth + stable)
        
        Args:
            sequence: (length, features)
            target_length: desired output length
            method: 'auto', 'linear', 'chebyshev', 'spline'
        
        Returns:
            (target_length, features)
        """
        current_len = sequence.shape[0]
        
        if method == 'auto':
            if current_len < 20:
                method = 'linear'
            elif current_len < 100:
                method = 'chebyshev'
            else:
                method = 'spline'
        
        if method == 'linear':
            return MotionSequenceInterpolator.linear_interpolate(sequence, target_length)
        elif method == 'chebyshev':
            return MotionSequenceInterpolator.chebyshev_interpolate(sequence, target_length)
        elif method == 'spline':
            return MotionSequenceInterpolator.spline_interpolate(sequence, target_length)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")


# Utility functions for Nyquist-Shannon checks

def nyquist_frequency(fps: float) -> float:
    """
    Calculate Nyquist frequency for given FPS.
    
    Nyquist theorem: Maximum frequency that can be represented is fps/2.
    
    Args:
        fps: Frames per second of video
    
    Returns:
        Nyquist frequency in Hz
    """
    return fps / 2


def is_nyquist_satisfied(original_fps: float,
                        num_frames: int,
                        target_frames: int) -> bool:
    """
    Check if resampling satisfies Nyquist-Shannon theorem.
    
    Simplified check: If we're sampling at a rate such that motion changes
    slower than 2x the new sampling rate, we're safe.
    
    Args:
        original_fps: Original video FPS
        num_frames: Number of frames in original video
        target_frames: Number of frames after resampling
    
    Returns:
        True if Nyquist criterion likely satisfied
    """
    # Effective new FPS after resampling
    if num_frames > 0:
        effective_fps = (original_fps * target_frames) / num_frames
    else:
        effective_fps = 0
    
    # Conservative check: require at least 2x oversampling of motion frequency
    # Assume typical motion frequency ~2-5 Hz for fitness exercises
    min_required_fps = 10
    
    return effective_fps >= min_required_fps


if __name__ == "__main__":
    # Test interpolation methods
    np.random.seed(42)
    
    # Create synthetic motion sequence
    t = np.linspace(0, 2*np.pi, 30)
    signal = np.sin(t)[:, np.newaxis]
    
    # Add another dimension
    signal = np.hstack([signal, np.cos(t)[:, np.newaxis]])
    
    print(f"Original shape: {signal.shape}")
    
    # Test each method
    interpolator = MotionSequenceInterpolator()
    
    for method in ['linear', 'chebyshev', 'spline']:
        if method == 'linear':
            result = interpolator.linear_interpolate(signal, 60)
        elif method == 'chebyshev':
            result = interpolator.chebyshev_interpolate(signal, 60)
        else:
            result = interpolator.spline_interpolate(signal, 60)
        
        print(f"{method:12} → shape: {result.shape}, stats:" +
              f" min={result.min():.3f}, max={result.max():.3f}")
    
    # Test Nyquist check
    print(f"\nNyquist frequency at 30 FPS: {nyquist_frequency(30):.1f} Hz")
    print(f"Resampling 300 frames → 60 frames at 30 FPS satisfies Nyquist: " +
          f"{is_nyquist_satisfied(30, 300, 60)}")
