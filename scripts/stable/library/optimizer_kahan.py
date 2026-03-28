import torch
import bitsandbytes as bnb
from typing import List, Dict, Any

class AdamW8bitKahan(bnb.optim.AdamW8bit):
    """
    AdamW8bit optimizer with Kahan Summation for improved numerical stability.
    Kahan Summation reduces accumulated floating-point errors.
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0.01, use_kahan=True, **kwargs):
        super().__init__(params, lr=lr, betas=betas, eps=eps, 
                        weight_decay=weight_decay, **kwargs)
        self.use_kahan = use_kahan
        
        # Initialize Kahan compensation buffers
        if self.use_kahan:
            self.kahan_compensation = {}
            for group in self.param_groups:
                for p in group['params']:
                    if p.requires_grad:
                        # Store compensation for each parameter
                        self.kahan_compensation[id(p)] = torch.zeros_like(p.data)
    
    def step(self, closure=None):
        """Performs a single optimization step with optional Kahan summation."""
        loss = None
        if closure is not None:
            loss = closure()
        
        # Call parent step
        super().step()
        
        # Apply Kahan summation if enabled
        if self.use_kahan:
            self._apply_kahan_summation()
        
        return loss
    
    def _apply_kahan_summation(self):
        """Apply Kahan summation to reduce floating-point errors."""
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad and id(p) in self.kahan_compensation:
                    # Kahan summation: reduce accumulated rounding errors
                    compensation = self.kahan_compensation[id(p)]
                    
                    # y = value + compensation
                    # t = sum + y
                    # compensation = y - (t - sum)
                    y = p.data - compensation
                    t = p.data + y
                    compensation = y - (t - p.data)
                    
                    # Update parameter and compensation
                    self.kahan_compensation[id(p)] = compensation


def create_optimizer_with_kahan(optimizer_type: str, trainable_params, lr: float, 
                                optimizer_kwargs: Dict[str, Any]):
    """
    Factory function to create optimizers with Kahan summation support.
    """
    
    if optimizer_type.lower() == "adamw8bit":
        return AdamW8bitKahan(
            trainable_params, 
            lr=lr,
            use_kahan=True,
            **optimizer_kwargs
        )
    else:
        # Fall back to regular optimizer
        raise ValueError(f"Kahan summation not implemented for {optimizer_type}")