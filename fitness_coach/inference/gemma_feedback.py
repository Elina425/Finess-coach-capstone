"""
Gemma-X Feedback Generator

Uses the Gemma language model to generate natural language feedback for exercises.

Workflow:
1. Model predicts exercise class and quality score
2. Extract problematic joints/angles from motion analysis
3. Build prompt with exercise context and metrics
4. Generate feedback using Gemma
5. Return actionable coaching tips

Gemma Model:
- google/gemma-2b-it: 2B parameters, lightweight, suitable for edge
- google/gemma-7b-it: 7B parameters, better quality, needs more compute
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict, List
import json


class GemmaFeedbackGenerator:
    """
    Generate natural language exercise feedback using Gemma language model.
    
    Features:
    - Configurable model size (2B or 7B)
    - Multi-turn conversation support
    - Context-aware feedback generation
    - Structured prompt engineering
    """
    
    # Model options
    AVAILABLE_MODELS = {
        'gemma-2b': 'google/gemma-2b-it',
        'gemma-7b': 'google/gemma-7b-it',
        'gemma-2-2b': 'google/gemma-2-2b-it',
        'gemma-2-9b': 'google/gemma-2-9b-it',
    }
    
    def __init__(self,
                 model_name: str = 'gemma-2b',
                 device: str = 'cpu',
                 dtype: torch.dtype = None,
                 max_new_tokens: int = 150,
                 temperature: float = 0.7,
                 top_p: float = 0.9):
        """
        Initialize Gemma feedback generator.
        
        Args:
            model_name: Model identifier (see AVAILABLE_MODELS)
            device: 'cpu', 'cuda', or 'mps' (for Mac)
            dtype: Torch dtype (auto-detect if None)
            max_new_tokens: Maximum length of generated feedback
            temperature: Sampling temperature (0 = deterministic, higher = more varied)
            top_p: Nucleus sampling parameter
        """
        
        # Resolve model name
        if model_name in self.AVAILABLE_MODELS:
            self.model_id = self.AVAILABLE_MODELS[model_name]
            self.model_name = model_name
        else:
            self.model_id = model_name  # Assume full HuggingFace ID
            self.model_name = model_name
        
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Auto-detect dtype
        if dtype is None:
            if device == 'cuda':
                dtype = torch.float16  # Use half precision on GPU
            else:
                dtype = torch.float32  # Use full precision on CPU
        
        self.dtype = dtype
        
        print(f"Loading Gemma model: {self.model_id}")
        print(f"  Device: {device}")
        print(f"  Dtype: {dtype}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map=device,
                low_cpu_mem_usage=True
            )
            self.model.eval()
            
            print(f"✓ Model loaded successfully")
            print(f"  Model size: {sum(p.numel() for p in self.model.parameters()):,} parameters")
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            print(f"  Make sure to: pip install transformers huggingface-hub")
            raise
    
    def generate_feedback(self,
                         exercise_class: str,
                         quality_score: float,
                         problematic_joints: Optional[List[str]] = None,
                         biomechanics_dict: Optional[Dict[str, float]] = None,
                         additional_context: Optional[str] = None) -> str:
        """
        Generate personalized exercise feedback.
        
        Args:
            exercise_class: Name of exercise (e.g., 'squat', 'biceps curl')
            quality_score: Form quality [0, 5]
            problematic_joints: List of joints with poor form
            biomechanics_dict: Dict of joint angles/measurements
            additional_context: Any other relevant context
        
        Returns:
            Feedback: Natural language coaching tips
        """
        
        # Build prompt
        prompt = self._build_prompt(
            exercise_class=exercise_class,
            quality_score=quality_score,
            problematic_joints=problematic_joints,
            biomechanics_dict=biomechanics_dict,
            additional_context=additional_context
        )
        
        # Generate response
        feedback = self._generate(prompt)
        
        return feedback
    
    def _build_prompt(self,
                     exercise_class: str,
                     quality_score: float,
                     problematic_joints: Optional[List[str]] = None,
                     biomechanics_dict: Optional[Dict[str, float]] = None,
                     additional_context: Optional[str] = None) -> str:
        """
        Build structured prompt for Gemma.
        
        Prompt engineering strategy:
        1. Set context (you are a fitness coach)
        2. Provide exercise info  
        3. Include performance metrics
        4. List problematic joints
        5. Request specific, actionable advice
        """
        
        prompt = "You are a professional fitness coach analyzing exercise performance.\n\n"
        
        # About the exercise
        prompt += f"EXERCISE PERFORMED: {exercise_class.upper()}\n"
        
        # Quality assessment
        prompt += f"FORM QUALITY SCORE: {quality_score:.1f}/5.0\n"
        
        if quality_score < 2.0:
            prompt += "ASSESSMENT: Poor form. Significant corrections needed.\n"
        elif quality_score < 3.0:
            prompt += "ASSESSMENT: Fair form. Several areas for improvement.\n"
        elif quality_score < 4.0:
            prompt += "ASSESSMENT: Good form. Minor refinements recommended.\n"
        else:
            prompt += "ASSESSMENT: Excellent form. Very few adjustments needed.\n"
        
        # Problematic areas
        if problematic_joints:
            prompt += f"\nPROBLEMATIC AREAS:\n"
            for joint in problematic_joints[:3]:  # Top 3 issues
                prompt += f"  - {joint}\n"
        
        # Biomechanical measurements
        if biomechanics_dict:
            prompt += f"\nKEY MEASUREMENTS:\n"
            for i, (joint, value) in enumerate(biomechanics_dict.items()):
                if i >= 3:  # Show top 3 measurements
                    break
                prompt += f"  - {joint}: {value:.1f}°\n"
        
        # Additional context
        if additional_context:
            prompt += f"\nADDITIONAL NOTES: {additional_context}\n"
        
        # Request specific feedback
        prompt += f"\nPROVIDE SPECIFIC COACHING TIPS:\n"
        prompt += "Based on the analysis above, provide 2-3 specific, actionable tips "
        prompt += f"to improve the {exercise_class} form. Focus on the problematic areas. "
        prompt += "Be concise and practical.\n\n"
        prompt += "COACHING FEEDBACK:"
        
        return prompt
    
    def _generate(self, prompt: str) -> str:
        """
        Generate text using Gemma.
        
        Args:
            prompt: Input prompt
        
        Returns:
            Generated text
        """
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract generated text (remove prompt)
        feedback = full_text.replace(prompt, "").strip()
        
        # Clean up if needed
        if feedback.startswith("Coaching Feedback:"):
            feedback = feedback.replace("Coaching Feedback:", "").strip()
        
        return feedback
    
    def interactive_feedback_session(self,
                                     exercise_class: str,
                                     num_generations: int = 3) -> List[str]:
        """
        Generate multiple variations of feedback for the same exercise.
        
        Useful for exploring different coaching approaches.
        
        Args:
            exercise_class: The exercise to get feedback for
            num_generations: Number of variations to generate
        
        Returns:
            List of feedback variations
        """
        
        feedbacks = []
        
        print(f"Generating {num_generations} variations of feedback for {exercise_class}...\n")
        
        for i in range(num_generations):
            # Use different temperatures for variety
            old_temp = self.temperature
            self.temperature = 0.5 + (i * 0.3)  # Increase variation
            
            feedback = self.generate_feedback(
                exercise_class=exercise_class,
                quality_score=3.2,  # Demo quality score
                problematic_joints=['hip', 'knee'],
                biomechanics_dict={'knee_angle': 75, 'hip_angle': 85}
            )
            
            feedbacks.append(feedback)
            
            self.temperature = old_temp
            
            print(f"Variation {i+1}:")
            print(feedback)
            print()
        
        return feedbacks


# Standalone utilities

def exercise_analysis_to_feedback(prediction_dict: Dict) -> str:
    """
    Convert model prediction dict to feedback string.
    
    Args:
        prediction_dict: {
            'exercise': str,
            'quality_score': float,
            'confidence': float,
            'problematic_angles': list,
            'measurements': dict
        }
    
    Returns:
        Feedback string
    """
    
    generator = GemmaFeedbackGenerator(model_name='gemma-2b')
    
    feedback = generator.generate_feedback(
        exercise_class=prediction_dict.get('exercise', 'exercise'),
        quality_score=prediction_dict.get('quality_score', 3.0),
        problematic_joints=prediction_dict.get('problematic_angles'),
        biomechanics_dict=prediction_dict.get('measurements')
    )
    
    return feedback


if __name__ == "__main__":
    # Example usage
    print("Gemma Feedback Generator Demo")
    print("=" * 60)
    
    try:
        # Initialize generator
        generator = GemmaFeedbackGenerator(
            model_name='gemma-2b',
            device='cpu',  # Use 'cuda' if available
            temperature=0.7,
            top_p=0.9
        )
        
        # Generate feedback for a squat
        print("\nGenerating feedback for squat exercise...\n")
        feedback = generator.generate_feedback(
            exercise_class='squat',
            quality_score=3.2,
            problematic_joints=['hip', 'knee', 'ankle'],
            biomechanics_dict={
                'hip_angle': 85,
                'knee_angle': 75,
                'ankle_angle': 70,
                'back_angle': 45
            },
            additional_context='Subject has tight calves'
        )
        
        print("FEEDBACK:")
        print(feedback)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Gemma model requires:")
        print("  - transformers >= 4.30")
        print("  - torch >= 2.0")
        print("  - huggingface_hub")
        print("\nInstall with: pip install transformers torch huggingface_hub")
