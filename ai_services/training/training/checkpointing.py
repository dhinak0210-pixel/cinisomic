"""
Checkpoint management utilities.
"""

import os
import torch
import shutil
from typing import Dict, Optional, Any
from pathlib import Path
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manager for saving and loading checkpoints.
    
    Args:
        save_dir: Directory to save checkpoints
        save_interval: Steps between checkpoints
        keep_last: Number of recent checkpoints to keep
    """
    
    def __init__(
        self,
        save_dir: str,
        save_interval: int = 1000,
        keep_last: int = 5,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_interval = save_interval
        self.keep_last = keep_last
        
        # Tracking
        self.best_score = float('inf') if True else float('-inf')
        self.checkpoint_count = 0
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        step: int,
        score: float,
        is_best: bool = False,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Save checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch
            step: Current step
            score: Validation score
            is_best: Whether this is the best checkpoint
            metadata: Additional metadata
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'score': score,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metadata': metadata or {},
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'checkpoint_epoch_{epoch:04d}_step_{step:06d}_{timestamp}.pth'
        save_path = self.save_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint, save_path)
        self.checkpoint_count += 1
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / 'best.pth'
            shutil.copy(save_path, best_path)
            logger.info(f"New best checkpoint saved: {score:.4f}")
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        logger.info(f"Checkpoint saved: {save_path}")
        return str(save_path)
    
    def load(
        self,
        path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            path: Path to checkpoint
            model: PyTorch model
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            strict: Strict loading of weights
            
        Returns:
            Checkpoint metadata
        """
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return {}
        
        checkpoint = torch.load(path, map_location='cpu')
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Load optimizer
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded: {path}")
        
        # Remove large state dicts from memory
        del checkpoint['model_state_dict']
        if 'optimizer_state_dict' in checkpoint:
            del checkpoint['optimizer_state_dict']
        if 'scheduler_state_dict' in checkpoint:
            del checkpoint['scheduler_state_dict']
        
        return checkpoint
    
    def load_partial(
        self,
        path: str,
        model: torch.nn.Module,
        exclude: Optional[list] = None,
    ) -> None:
        """
        Load partial checkpoint (for transfer learning).
        
        Args:
            path: Path to checkpoint
            model: PyTorch model
            exclude: List of parameter names to exclude
        """
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return
        
        checkpoint = torch.load(path, map_location='cpu')
        model_state = checkpoint.get('model_state_dict', {})
        
        if exclude is None:
            exclude = []
        
        # Load matching weights
        model_dict = model.state_dict()
        for name, param in model_state.items():
            if name in model_dict and name not in exclude:
                if param.shape == model_dict[name].shape:
                    model_dict[name] = param
        
        model.load_state_dict(model_dict, strict=False)
        logger.info(f"Partial checkpoint loaded: {path}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        if self.keep_last <= 0:
            return
        
        # Get all checkpoints sorted by time
        checkpoints = sorted(
            self.save_dir.glob('checkpoint_*.pth'),
            key=lambda p: p.stat().st_mtime,
        )
        
        # Keep last N checkpoints
        checkpoints_to_remove = checkpoints[:-self.keep_last]
        
        for checkpoint in checkpoints_to_remove:
            checkpoint.unlink()
            logger.debug(f"Removed old checkpoint: {checkpoint}")
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        checkpoints = sorted(
            self.save_dir.glob('checkpoint_*.pth'),
            key=lambda p: p.stat().st_mtime,
        )
        
        if checkpoints:
            return str(checkpoints[-1])
        return None
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint."""
        best_path = self.save_dir / 'best.pth'
        if best_path.exists():
            return str(best_path)
        return None
    
    def list_checkpoints(self) -> list:
        """List all checkpoints."""
        return sorted(self.save_dir.glob('checkpoint_*.pth'))

