# Cinematic Music Generation - Training Infrastructure

A comprehensive training infrastructure for large-scale deep learning system for cinematic music and song generation.

## Overview

This training module provides everything needed to train transformer-based music generation models, voice cloning systems, and multilingual lyrics generators.

## Features

- **Transformer-based Music Generation**: Music Transformer with relative attention and emotion conditioning
- **Voice Cloning**: FastSpeech 2 with HiFi-GAN vocoder for natural voice synthesis
- **Multilingual Lyrics Generation**: GPT-based model with emotion, genre, and scene context conditioning
- **Distributed Training**: Multi-GPU training with PyTorch Distributed
- **Mixed Precision**: FP16 training for memory efficiency
- **Data Augmentation**: 1000+ variations through audio transformations
- **Bias Reduction**: Built-in tools for balanced, diverse datasets

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
training/
├── requirements.txt       # Dependencies
├── config.yaml           # Main configuration
├── configs/              # Configuration classes
│   ├── __init__.py
│   ├── model_config.py   # Model hyperparameters
│   └── training_config.py # Training settings
├── data/                 # Data pipeline
│   ├── __init__.py
│   ├── dataset.py        # Dataset classes
│   ├── preprocessing.py  # Audio preprocessing
│   ├── augmentation.py   # Data augmentation
│   ├── cleaning.py       # Data cleaning
│   └── loaders.py        # Data loaders
├── models/               # Model architectures
│   ├── __init__.py
│   ├── music_transformer.py  # Music generation
│   ├── fastspeech.py     # Voice cloning
│   └── lyrics_generator.py  # Lyrics generation
├── training/             # Training utilities
│   ├── __init__.py
│   ├── trainer.py        # Training loops
│   ├── distributed.py    # Distributed training
│   ├── optimizers.py     # Optimizers & schedulers
│   ├── ema.py            # Exponential moving average
│   └── checkpointing.py  # Checkpoint management
├── train_music.py        # Music training script
├── train_voice.py        # Voice training script
└── train_lyrics.py       # Lyrics training script
```

## Quick Start

### 1. Prepare Your Data

Organize your data in the following structure:

```
data/
├── music/
│   ├── track1.wav
│   ├── track1.json  (metadata)
│   ├── track2.wav
│   └── ...
├── voice/
│   ├── speaker1/
│   │   ├── audio1.wav
│   │   ├── audio1.txt (transcript)
│   │   └── ...
│   └── speaker2/
│       └── ...
└── lyrics/
    ├── en/
    │   ├── song1.txt
    │   └── song1.json
    └── es/
        └── ...
```

### 2. Configure Training

Edit `config.yaml` to match your setup:

```yaml
data:
  batch_size: 8
  sample_rate: 44100
  segment_duration: 10.0

music_model:
  num_layers: 12
  num_heads: 12
  embed_dim: 768

training:
  epochs: 100
  learning_rate: 1e-4
  amp: true
```

### 3. Start Training

**Music Generation:**
```bash
python train_music.py --config config.yaml --data_dir ./data/music
```

**Voice Cloning:**
```bash
python train_voice.py --config config.yaml --data_dir ./data/voice
```

**Lyrics Generation:**
```bash
python train_lyrics.py --config config.yaml --data_dir ./data/lyrics
```

### 4. Resume Training

```bash
python train_music.py --resume ./work_dir/checkpoints/best.pth
```

## Configuration Options

### Model Configuration

#### Music Transformer
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_layers` | 12 | Number of transformer layers |
| `num_heads` | 12 | Number of attention heads |
| `embed_dim` | 768 | Embedding dimension |
| `dropout` | 0.1 | Dropout probability |
| `emotion_conditioning` | true | Enable emotion conditioning |

#### FastSpeech 2 (Voice)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `encoder_num_layers` | 4 | Encoder layers |
| `decoder_num_layers` | 4 | Decoder layers |
| `num_speakers` | 100 | Number of speakers |
| `num_mels` | 80 | Mel spectrogram bins |

#### Lyrics Generator
| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 50257 | Vocabulary size |
| `num_layers` | 12 | Transformer layers |
| `languages` | ["en", "es", "fr", "de"] | Supported languages |
| `emotion_conditioning` | true | Enable emotion labels |

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Training epochs |
| `batch_size` | 8 | Batch size per GPU |
| `learning_rate` | 1e-4 | Initial learning rate |
| `optimizer` | "adamw" | Optimizer type |
| `scheduler` | "cosine_warmup" | LR scheduler |
| `warmup_steps` | 10000 | Warmup steps |
| `amp` | true | Enable mixed precision |
| `gradient_clip_val` | 1.0 | Gradient clipping |

## Data Augmentation

The system supports extensive audio augmentation:

- **Tempo Variation**: ±15% speed changes
- **Pitch Shift**: ±3 semitones
- **Dynamic Range**: Compression
- **Reverb**: Room simulation
- **Noise Injection**: White, pink, brown noise
- **EQ**: Frequency band adjustments
- **Stereo Width**: Stereo enhancement

Generate 1000+ variations per sample:

```python
from data.augmentation import create_augmentation_config, VariationGenerator

config = create_augmentation_config(variation_count=1000)
generator = VariationGenerator(config, sample_rate=44100)

variations = generator.generate_all_variations(audio)
```

## Distributed Training

Run on multiple GPUs:

```bash
# Single node, multi-GPU
python -m torch.distributed.launch --nproc_per_node=8 train_music.py

# Multi-node
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=xxx.xxx.xxx.xxx \
    train_music.py
```

## Monitoring

### TensorBoard
```bash
tensorboard --logdir ./work_dir/logs/tensorboard
```

### Weights & Biases
```yaml
# config.yaml
logging:
  logger: "wandb"
  wandb:
    project: "cinematic-music-generation"
    entity: "your-username"
```

## Checkpoints

Checkpoints are saved to `./work_dir/checkpoints/` with:
- Model weights
- Optimizer state
- Learning rate scheduler state
- Training metrics

## Best Practices

1. **Start Small**: Begin with smaller model settings for debugging
2. **Monitor Loss**: Watch for NaN or exploding gradients
3. **Save Frequently**: Enable frequent checkpointing during long runs
4. **Use Mixed Precision**: FP16 training reduces memory by ~50%
5. **Balance Data**: Use bias reduction for diverse training

## Troubleshooting

### Out of Memory
- Reduce batch size
- Enable gradient accumulation
- Use smaller model dimensions

### Slow Training
- Increase `num_workers` for data loading
- Use SSD storage for data
- Enable mixed precision training

### Poor Quality
- Increase training data
- Add more augmentation
- Adjust model size

## License

MIT License - See LICENSE file for details.

## Citation

If you use this code, please cite:

```bibtex
@misc{cinematic-music-2024,
  title = {Cinematic Music Generation Training Infrastructure},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/cinematic-music-platform}
}
```

