#!/usr/bin/env python3
"""
Main training script for Voice Cloning (FastSpeech 2).
Usage: python train_voice.py --config config.yaml
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

from training.configs.model_config import VoiceModelConfig
from training.configs.training_config import TrainingConfig
from training.models.fastspeech import FastSpeech2
from training.models.fastspeech import HiFiGAN
from training.data import VoiceDataset, create_data_loaders
from training.training import (
    create_optimizer, 
    create_scheduler, 
    VoiceTrainer,
    MixedPrecisionManager,
)
from training.training.checkpointing import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Voice Cloning Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='./data/voice',
                        help='Directory containing voice data')
    parser.add_argument('--work_dir', type=str, default='./work_dir/voice',
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
    model_config = VoiceModelConfig(
        encoder_embed_dim=config['voice_model']['encoder_embed_dim'],
        encoder_num_layers=config['voice_model']['encoder_num_layers'],
        decoder_embed_dim=config['voice_model']['decoder_embed_dim'],
        decoder_num_layers=config['voice_model']['decoder_num_layers'],
        num_speakers=config['voice_model']['num_speakers'],
        learning_rate=config['voice_model']['learning_rate'],
    )
    
    # Create model
    model = FastSpeech2(model_config)
    logger.info(f"FastSpeech2 created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create vocoder
    vocoder = HiFiGAN(model_config)
    
    # Create datasets
    train_dataset = VoiceDataset(
        data_root=args.data_dir,
        split='train',
        max_samples=config['data'].get('max_samples'),
    )
    
    val_dataset = VoiceDataset(
        data_root=args.data_dir,
        split='val',
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
        learning_rate=config['voice_model']['learning_rate'],
        weight_decay=config['voice_model']['weight_decay'],
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
    trainer = VoiceTrainer(
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
    
    logger.info("Voice training completed!")
    logger.info(f"Final mel loss: {history['train_loss'][-1]:.4f}")


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    main()

