#!/usr/bin/env python3
"""
Main training script for Lyrics Generation.
Usage: python train_lyrics.py --config config.yaml
"""

import os
import sys
import argparse
import yaml
import torch
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from training.configs.model_config import LyricsModelConfig
from training.configs.training_config import TrainingConfig
from training.models import LyricsGenerator
from training.data import LyricsDataset, create_data_loaders
from training.training import (
    create_optimizer, 
    create_scheduler, 
    LyricsTrainer,
    MixedPrecisionManager,
)
from training.training.checkpointing import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Lyrics Generator')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='./data/lyrics',
                        help='Directory containing lyrics data')
    parser.add_argument('--work_dir', type=str, default='./work_dir/lyrics',
                        help='Working directory for outputs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device ID')
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    config = load_config(args.config)
    
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model config
    model_config = LyricsGeneratorConfig(
        vocab_size=config['lyrics_model']['vocab_size'],
        num_layers=config['lyrics_model']['num_layers'],
        num_heads=config['lyrics_model']['num_heads'],
        embed_dim=config['lyrics_model']['embed_dim'],
        ff_dim=config['lyrics_model']['ff_dim'],
        max_length=config['lyrics_model']['max_length'],
        languages=config['lyrics_model']['languages'],
        num_emotions=config['lyrics_model']['num_emotions'],
        num_genres=config['lyrics_model']['num_genres'],
        learning_rate=config['lyrics_model']['learning_rate'],
    )
    
    # Create model
    model = LyricsGenerator(model_config)
    logger.info(f"LyricsGenerator created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create datasets
    train_dataset = LyricsDataset(
        data_root=args.data_dir,
        split='train',
        languages=config['lyrics_model']['languages'],
        max_samples=config['data'].get('max_samples'),
    )
    
    val_dataset = LyricsDataset(
        data_root=args.data_dir,
        split='val',
        languages=config['lyrics_model']['languages'],
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
        learning_rate=config['lyrics_model']['learning_rate'],
        weight_decay=config['lyrics_model']['weight_decay'],
    )
    
    # Create scheduler
    num_training_steps = len(loaders['train']) * config['training']['epochs']
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type=config['training']['scheduler'],
        num_training_steps=num_training_steps,
        num_warmup_steps=config['training']['warmup_steps'],
    )
    
    # Create trainer
    trainer = LyricsTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config['training'],
        device=device,
        work_dir=str(work_dir),
    )
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    history = trainer.fit(
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        epochs=config['training']['epochs'],
    )
    
    logger.info("Lyrics training completed!")
    logger.info(f"Final loss: {history['train_loss'][-1]:.4f}")


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    main()

