"""
Distributed training utilities for scalable training.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DistributedManager:
    """
    Manager for distributed training operations.
    
    Args:
        backend: Distributed backend ('nccl', 'gloo', etc.)
        init_method: Initialization method ('env', 'tcp', 'file')
        world_size: Number of processes
        rank: Current process rank
    """
    
    def __init__(
        self,
        backend: str = 'nccl',
        init_method: str = 'env',
        world_size: int = -1,
        rank: int = 0,
    ):
        self.backend = backend
        self.init_method = init_method
        self.world_size = world_size
        self.rank = rank
        
        self._initialized = False
    
    def setup(self):
        """Initialize distributed training."""
        if self.world_size <= 0:
            # Auto-detect world size
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        self.rank = int(os.environ.get('RANK', 0))
        
        if self.world_size > 1 and not self._initialized:
            # Initialize process group
            dist.init_process_group(
                backend=self.backend,
                init_method=self.init_method,
                world_size=self.world_size,
                rank=self.rank,
            )
            
            # Set CUDA device
            torch.cuda.set_device(self.rank)
            
            self._initialized = True
            logger.info(f"Distributed training initialized: {self.world_size} processes, rank {self.rank}")
    
    def cleanup(self):
        """Clean up distributed training."""
        if self._initialized:
            dist.destroy_process_group()
            self._initialized = False
            logger.info("Distributed training cleaned up")
    
    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.rank == 0
    
    @property
    def is_distributed(self) -> bool:
        """Check if distributed training is enabled."""
        return self.world_size > 1
    
    def barrier(self):
        """Synchronize all processes."""
        if self.is_distributed:
            dist.barrier()
    
    def all_reduce(self, tensor: torch.Tensor, op: str = 'sum'):
        """All-reduce a tensor across all processes."""
        if self.is_distributed:
            op_enum = dist.ReduceOp.SUM if op == 'sum' else dist.ReduceOp.AVG
            dist.all_reduce(tensor, op=op_enum)
    
    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensors from all processes."""
        if not self.is_distributed:
            return tensor
        
        output = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather_into(output, tensor)
        return torch.cat(output, dim=0)


class MixedPrecisionManager:
    """
    Manager for mixed precision training.
    Uses PyTorch's native AMP (Automatic Mixed Precision).
    
    Args:
        enabled: Enable mixed precision
        opt_level: Optimization level ('O0', 'O1', 'O2', 'O3')
        cache_enabled: Enable gradient caching
    """
    
    def __init__(
        self,
        enabled: bool = True,
        opt_level: str = 'O1',
        cache_enabled: bool = True,
    ):
        self.enabled = enabled
        self.opt_level = opt_level
        self.cache_enabled = cache_enabled
        
        self.scaler = None
        
        if enabled:
            self.scaler = torch.cuda.amp.GradScaler(
                enabled=True,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000,
                enabled=cache_enabled,
            )
    
    def __enter__(self):
        """Context manager for forward pass."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        return False
    
    def autocast(self):
        """Get autocast context manager."""
        return torch.cuda.amp.autocast(
            enabled=self.enabled,
            dtype=torch.float16,
        )
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for gradient accumulation.
        
        Args:
            loss: Loss tensor
            
        Returns:
            Scaled loss
        """
        if self.enabled and self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def step(self, optimizer: torch.optim.Optimizer):
        """
        Step optimizer with gradient scaling.
        
        Args:
            optimizer: Optimizer to step
        """
        if self.enabled and self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def update(self):
        """Update the scaler."""
        if self.scaler is not None:
            self.scaler.update()


def setup_distributed_training(
    backend: str = 'nccl',
    local_rank: int = 0,
) -> tuple:
    """
    Set up distributed training environment.
    
    Args:
        backend: Distributed backend
        local_rank: Local GPU rank
        
    Returns:
        Tuple of (device, distributed_manager)
    """
    # Set environment variables
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = os.environ.get('RANK', '0')
    os.environ['WORLD_SIZE'] = os.environ.get('WORLD_SIZE', '1')
    
    # Create distributed manager
    dist_manager = DistributedManager(
        backend=backend,
        init_method='env',
    )
    
    # Setup
    dist_manager.setup()
    
    # Set device
    device = torch.device(f'cuda:{local_rank}')
    
    return device, dist_manager

