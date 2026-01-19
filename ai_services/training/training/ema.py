"""
Exponential Moving Average (EMA) utilities.
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ExponentialMovingAverage:
    """
    Exponential Moving Average for model weights.
    Provides more stable training and better generalization.
    
    Args:
        model: PyTorch model
        decay: EMA decay rate
        update_every: Update frequency
        device: Device to store EMA weights
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_every: int = 10,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.decay = decay
        self.update_every = update_every
        self.device = device
        
        # Create shadow model
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow model with current weights
        self._initialize_shadow()
        
        logger.info(f"EMA initialized with decay={decay}, update_every={update_every}")
    
    def _initialize_shadow(self):
        """Initialize shadow model with current weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                if self.device is not None:
                    self.shadow[name] = self.shadow[name].to(self.device)
    
    def update(self, step: Optional[int] = None):
        """
        Update EMA weights.
        
        Args:
            step: Current training step
        """
        # Update based on frequency
        if step is not None and step % self.update_every != 0:
            return
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (
                    (1.0 - self.decay) * param.data +
                    self.decay * self.shadow[name]
                )
                self.shadow[name] = new_average.detach().clone()
    
    def apply_shadow(self):
        """
        Apply EMA weights to model.
        Used for evaluation or inference.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """
        Restore original weights.
        Used after evaluation.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    @torch.no_grad()
    def to(self, device: torch.device):
        """Move EMA weights to device."""
        for name in self.shadow:
            self.shadow[name] = self.shadow[name].to(device)
        self.device = device
    
    def state_dict(self) -> dict:
        """Get EMA state."""
        return {
            'shadow': self.shadow,
            'decay': self.decay,
            'update_every': self.update_every,
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load EMA state."""
        self.shadow = state_dict['shadow']
        self.decay = state_dict.get('decay', self.decay)
        self.update_every = state_dict.get('update_every', self.update_every)

