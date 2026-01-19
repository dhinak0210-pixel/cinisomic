"""
Optimizer and scheduler utilities.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, LinearLR
from typing import Dict, Optional, Type
import logging

logger = logging.getLogger(__name__)


def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = 'adamw',
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    momentum: float = 0.9,
) -> optim.Optimizer:
    """
    Create optimizer for model training.
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd', 'rmsprop')
        learning_rate: Learning rate
        weight_decay: Weight decay
        betas: Adam beta parameters
        eps: Adam epsilon
        momentum: SGD momentum
        
    Returns:
        Configured optimizer
    """
    # Get model parameters
    params = model.parameters()
    
    # Filter frozen parameters
    params = [p for p in params if p.requires_grad]
    
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(
            params,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    elif optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(
            params,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(
            params,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif optimizer_type.lower() == 'rmsprop':
        optimizer = optim.RMSprop(
            params,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif optimizer_type.lower() == 'adagrad':
        optimizer = optim.Adagrad(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        logger.warning(f"Unknown optimizer type: {optimizer_type}, using AdamW")
        optimizer = optim.AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    
    logger.info(f"Created {optimizer_type} optimizer with lr={learning_rate}")
    return optimizer


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = 'cosine_warmup',
    num_training_steps: int = 100000,
    num_warmup_steps: int = 10000,
    min_lr: float = 1e-6,
    T_max: int = 100000,
) -> _LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Configured optimizer
        scheduler_type: Type of scheduler
        num_training_steps: Total training steps
        num_warmup_steps: Warmup steps
        min_lr: Minimum learning rate
        T_max: Period for cosine annealing
        
    Returns:
        Configured scheduler
    """
    if scheduler_type == 'cosine_warmup':
        # Cosine annealing with warmup
        from torch.optim.lr_scheduler import SequentialLR
        
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=num_warmup_steps,
        )
        
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - num_warmup_steps,
            eta_min=min_lr,
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[num_warmup_steps],
        )
        
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=min_lr,
        )
    
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=num_training_steps // 3,
            gamma=0.1,
        )
    
    elif scheduler_type == 'polynomial':
        scheduler = optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=num_training_steps,
            power=1.0,
        )
    
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=min_lr,
        )
    
    elif scheduler_type == 'none' or scheduler_type is None:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: 1.0,
        )
    
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, using constant")
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: 1.0,
        )
    
    logger.info(f"Created {scheduler_type} scheduler")
    return scheduler

