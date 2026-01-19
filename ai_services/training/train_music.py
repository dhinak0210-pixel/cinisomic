#!/usr/bin/env python3
"""
Main training script for Music Transformer.
Usage: python train_music.py --config config.yaml
"""

import os
import sys
import argparse
import yaml
import torch
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from training.configs.model_config import MusicModelConfig
from training.configs.training_config import TrainingConfig
from training.models import MusicTransformer
from training.data import MusicDataset, create_data_loaders
from training.training import (
    create_optimizer, 
    create_scheduler, 
    MusicTrainer,
    MixedPrecisionManager,
)
from training.training.checkpointing import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Music Transformer')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='./data/music',
                        help='Directory containing music data')
    parser.add_argument('--work_dir', type=str, default='./work_dir',
                        help='Working directory for outputs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device ID')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()
    
    # Set GPU
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line args
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['music_model']['learning_rate'] = args.learning_rate
    
    # Create work directory
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model configuration
    model_config = MusicModelConfig(
        num_layers=config['music_model']['num_layers'],
        num_heads=config['music_model']['num_heads'],
        embed_dim=config['music_model']['embed_dim'],
        ff_dim=config['music_model']['ff_dim'],
        dropout=config['music_model']['dropout'],
        learning_rate=config['music_model']['learning_rate'],
        max_sequence_length=config['music_model']['max_sequence_length'],
    )
    
    # Create model
    model = MusicTransformer(model_config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create datasets
    train_dataset = MusicDataset(
        data_root=args.data_dir,
        split='train',
        segment_duration=config['data']['segment_duration'],
        sample_rate=config['data']['sample_rate'],
        max_samples=config['data'].get('max_samples'),
    )
    
    val_dataset = MusicDataset(
        data_root=args.data_dir,
        split='val',
        segment_duration=config['data']['segment_duration'],
        sample_rate=config['data']['sample_rate'],
        max_samples=config['data'].get('max_samples'),
    )
    
    # Create data loaders
    loaders = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        distributed=False,
    )
    
    # Create optimizer
    optimizer = create_optimizer(
        model=model,
        optimizer_type=config['training']['optimizer'],
        learning_rate=config['music_model']['learning_rate'],
        weight_decay=config['music_model']['weight_decay'],
    )
    
    # Create scheduler
    num_training_steps = len(loaders['train']) * config['training']['epochs']
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type=config['training']['scheduler'],
        num_training_steps=num_training_steps,
        num_warmup_steps=config['training']['warmup_steps'],
    )
    
    # Create mixed precision manager
    amp_manager = MixedPrecisionManager(
        enabled=config['training']['amp'],
        opt_level=config['training']['amp_level'],
    )
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        save_dir=str(work_dir / 'checkpoints'),
        save_interval=config['checkpoint']['save_interval'],
        keep_last=config['checkpoint']['keep_last_checkpoints'],
    )
    
    # Create trainer
    trainer = MusicTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config['training'],
        device=device,
        work_dir=str(work_dir),
    )
    
    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train!
    history = trainer.fit(
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        epochs=config['training']['epochs'],
    )
    
    logger.info("Training completed!")
    logger.info(f"Final training loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss'][-1] is not None:
        logger.info(f"Final validation loss: {history['val_loss'][-1]:.4f}")


if __name__ == '__main__':
    main()

