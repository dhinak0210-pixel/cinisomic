"""
Base trainer class for all training scenarios.
Provides common training loop, logging, and validation utilities.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Optional, Any, List, Union
from pathlib import Path
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class BaseTrainer:
    """
    Base trainer class providing common training functionality.
    
    Args:
        model: PyTorch model to train
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        config: Training configuration
        device: Device to train on
        work_dir: Working directory for saving outputs
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        config: Dict,
        device: torch.device,
        work_dir: str = "./work_dir",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        # Set up working directory
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_metric = float('inf') if config.get('mode', 'min') == 'min' else float('-inf')
        
        # Logging
        self.log_interval = config.get('log_interval', 100)
        self.eval_interval = config.get('eval_interval', 1000)
        
        # Resume training
        self.start_epoch = 0
        self.start_step = 0
        
        # Move model to device
        self.model.to(self.device)
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Working directory: {self.work_dir}")
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Move batch to device
                batch = self._prepare_batch(batch)
                
                # Forward pass
                loss, metrics = self._training_step(batch)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.config.get('gradient_clip_val', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip_val']
                    )
                
                self.optimizer.step()
                
                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Logging
                if (batch_idx + 1) % self.log_interval == 0:
                    avg_loss = total_loss / num_batches
                    lr = self.optimizer.param_groups[0]['lr']
                    logger.info(
                        f"Epoch {self.current_epoch}, Batch {batch_idx + 1}/{len(dataloader)}, "
                        f"Loss: {avg_loss:.4f}, LR: {lr:.2e}"
                    )
                
                self.current_step += 1
                
            except Exception as e:
                logger.error(f"Error in training step {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return {
            'train_loss': avg_loss,
            'num_batches': num_batches,
        }
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                batch = self._prepare_batch(batch)
                loss, metrics = self._validation_step(batch)
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                logger.error(f"Error in validation step {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        return {
            'val_loss': avg_loss,
            'num_batches': num_batches,
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        callbacks: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            callbacks: List of callbacks
            
        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': [],
        }
        
        logger.info(f"Starting training for {epochs} epochs")
        start_time = time.time()
        
        for epoch in range(self.start_epoch, epochs):
            self.current_epoch = epoch
            
            epoch_start = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate if needed
            val_metrics = {}
            if val_loader is not None and (epoch + 1) % self.config.get('val_check_interval', 1) == 0:
                val_metrics = self.validate(val_loader)
                
                # Check for improvement
                val_loss = val_metrics.get('val_loss', float('inf'))
                mode = self.config.get('mode', 'min')
                
                is_best = (
                    (mode == 'min' and val_loss < self.best_metric) or
                    (mode == 'max' and val_loss > self.best_metric)
                )
                
                if is_best:
                    self.best_metric = val_loss
                    self._save_checkpoint('best')
                
                # Log validation
                logger.info(
                    f"Epoch {epoch}: Val Loss: {val_loss:.4f}, "
                    f"Best: {self.best_metric:.4f}"
                )
            
            epoch_time = time.time() - epoch_start
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('checkpoint_interval', 1) == 0:
                self._save_checkpoint(f'epoch_{epoch}')
            
            # Record history
            history['train_loss'].append(train_metrics['train_loss'])
            history['val_loss'].append(val_metrics.get('val_loss', None))
            history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            history['epoch_time'].append(epoch_time)
            
            logger.info(
                f"Epoch {epoch} completed in {epoch_time:.1f}s - "
                f"Train Loss: {train_metrics['train_loss']:.4f}"
            )
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/3600:.2f} hours")
        
        history['total_time'] = total_time
        
        return history
    
    def _training_step(self, batch: Dict) -> tuple:
        """
        Training step logic.
        Override in subclasses.
        
        Args:
            batch: Input batch
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        raise NotImplementedError
    
    def _validation_step(self, batch: Dict) -> tuple:
        """
        Validation step logic.
        Override in subclasses.
        
        Args:
            batch: Input batch
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        raise NotImplementedError
    
    def _prepare_batch(self, batch: Dict) -> Dict:
        """
        Prepare batch for training.
        Move tensors to device.
        
        Args:
            batch: Input batch
            
        Returns:
            Batch with tensors on correct device
        """
        prepared = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.device)
            else:
                prepared[key] = value
        return prepared
    
    def _save_checkpoint(self, name: str):
        """
        Save checkpoint.
        
        Args:
            name: Checkpoint name
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        save_path = self.work_dir / f'checkpoint_{name}.pth'
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, path: str):
        """
        Load checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.current_step = checkpoint.get('step', 0)
        self.best_metric = checkpoint.get('best_metric', self.best_metric)
        
        logger.info(f"Checkpoint loaded: {path}")
    
    def _log_metrics(self, metrics: Dict, prefix: str = ''):
        """
        Log metrics to console and file.
        
        Args:
            metrics: Dictionary of metrics
            prefix: Metric name prefix
        """
        log_dir = self.work_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log to file
        log_file = log_dir / f'{prefix}metrics.json'
        with open(log_file, 'a') as f:
            f.write(json.dumps({
                'step': self.current_step,
                'epoch': self.current_epoch,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat(),
            }) + '\n')
        
        # Log to console
        metrics_str = ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        logger.info(f"{prefix} Metrics: {metrics_str}")


class MusicTrainer(BaseTrainer):
    """
    Trainer for music generation models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        config: Dict,
        device: torch.device,
        work_dir: str = "./work_dir",
    ):
        super().__init__(model, optimizer, scheduler, config, device, work_dir)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def _training_step(self, batch: Dict) -> tuple:
        """
        Music generation training step.
        
        Args:
            batch: Input batch with audio and conditions
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        audio = batch['audio']  # (batch, channels, samples)
        emotion_ids = batch.get('emotion_ids')
        
        # Forward pass
        outputs = self.model(audio=audio, emotion_ids=emotion_ids)
        logits = outputs['logits']
        
        # Shift for autoregressive training
        target = audio[:, :, :-1].reshape(-1)
        logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        
        # Calculate loss
        loss = self.criterion(logits, target)
        
        # Calculate accuracy
        preds = logits.argmax(dim=-1)
        correct = (preds == target).float().mean()
        
        metrics = {
            'loss': loss.item(),
            'accuracy': correct.item(),
        }
        
        return loss, metrics
    
    def _validation_step(self, batch: Dict) -> tuple:
        """Validation step for music generation."""
        return self._training_step(batch)


class VoiceTrainer(BaseTrainer):
    """
    Trainer for voice cloning models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        config: Dict,
        device: torch.device,
        work_dir: str = "./work_dir",
    ):
        super().__init__(model, optimizer, scheduler, config, device, work_dir)
        
        # Loss functions
        self.mel_criterion = nn.MSELoss()
        self.duration_criterion = nn.MSELoss()
        self.pitch_criterion = nn.MSELoss()
    
    def _training_step(self, batch: Dict) -> tuple:
        """
        Voice cloning training step.
        
        Args:
            batch: Input batch with text, audio, and metadata
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        text = batch['text']  # (batch, seq_len)
        mel = batch['mel']  # (batch, num_mels, time)
        duration = batch.get('duration')
        pitch = batch.get('pitch')
        speaker_ids = batch.get('speaker_ids')
        
        # Forward pass
        outputs = self.model(
            text=text,
            duration=duration,
            pitch=pitch,
            speaker_ids=speaker_ids,
        )
        
        # Calculate losses
        mel_loss = self.mel_criterion(outputs['mel'], mel)
        
        total_loss = mel_loss
        
        metrics = {
            'mel_loss': mel_loss.item(),
            'total_loss': total_loss.item(),
        }
        
        # Add variance losses if available
        if 'duration_pred' in outputs and duration is not None:
            dur_loss = self.duration_criterion(outputs['duration_pred'], duration)
            metrics['duration_loss'] = dur_loss.item()
            total_loss = total_loss + dur_loss
        
        if 'pitch_pred' in outputs and pitch is not None:
            pitch_loss = self.pitch_criterion(outputs['pitch_pred'], pitch)
            metrics['pitch_loss'] = pitch_loss.item()
            total_loss = total_loss + pitch_loss
        
        return total_loss, metrics
    
    def _validation_step(self, batch: Dict) -> tuple:
        """Validation step for voice cloning."""
        return self._training_step(batch)


class LyricsTrainer(BaseTrainer):
    """
    Trainer for lyrics generation models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        config: Dict,
        device: torch.device,
        work_dir: str = "./work_dir",
    ):
        super().__init__(model, optimizer, scheduler, config, device, work_dir)
        
        self.criterion = nn.CrossEntropyLoss()
    
    def _training_step(self, batch: Dict) -> tuple:
        """
        Lyrics generation training step.
        
        Args:
            batch: Input batch with tokens and conditions
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        input_ids = batch['input_ids']  # (batch, seq_len)
        attention_mask = batch.get('attention_mask')
        language_ids = batch.get('language_ids')
        emotion_ids = batch.get('emotion_ids')
        genre_ids = batch.get('genre_ids')
        scene_features = batch.get('scene_features')
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            language_ids=language_ids,
            emotion_ids=emotion_ids,
            genre_ids=genre_ids,
            scene_features=scene_features,
        )
        
        logits = outputs['logits']
        
        # Shift for autoregressive training
        target = input_ids[:, 1:].reshape(-1)
        logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        
        # Calculate loss
        loss = self.criterion(logits, target)
        
        # Calculate accuracy
        preds = logits.argmax(dim=-1)
        correct = (preds == target).float().mean()
        
        metrics = {
            'loss': loss.item(),
            'accuracy': correct.item(),
        }
        
        return loss, metrics
    
    def _validation_step(self, batch: Dict) -> tuple:
        """Validation step for lyrics generation."""
        return self._training_step(batch)

