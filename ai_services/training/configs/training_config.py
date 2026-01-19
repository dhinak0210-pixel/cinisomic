"""
Training configuration classes for cinematic music generation.
Defines training loop, optimization, and logging settings.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


class OptimizerType(Enum):
    """Supported optimizer types."""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    ADAGRAD = "adagrad"
    RMSPROP = "rmsprop"


class SchedulerType(Enum):
    """Supported learning rate scheduler types."""
    STEP = "step"
    COSINE = "cosine"
    COSINE_WARMUP = "cosine_warmup"
    LINEAR_WARMUP = "linear_warmup"
    POLYNOMIAL = "polynomial"
    PLATEAU = "plateau"


class PrecisionType(Enum):
    """Supported precision types for training."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class BackendType(Enum):
    """Distributed training backend types."""
    NCCL = "nccl"
    GLOO = "gloo"
    MPI = "mpi"


@dataclass
class HardwareConfig:
    """Hardware configuration for training."""
    device: str = "cuda"
    precision: PrecisionType = PrecisionType.FP16
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    memory_format: str = "channels_last"  # channels_last, channels_first
    

@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    enabled: bool = True
    backend: BackendType = BackendType.NCCL
    init_method: str = "env"
    world_size: int = -1  # -1 for all available GPUs
    rank: int = 0
    local_rank: int = 0
    
    # Distributed optimizations
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    

@dataclass
class OptimizationConfig:
    """Optimization configuration."""
    # Optimizer settings
    optimizer: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # Learning rate scheduler
    scheduler: SchedulerType = SchedulerType.COSINE_WARMUP
    warmup_steps: int = 10000
    min_lr: float = 1e-6
    max_lr: float = 1e-3
    T_max: int = 100000  # For cosine scheduler
    factor: float = 0.5  # For plateau scheduler
    patience: int = 5  # For plateau scheduler
    
    # Gradient handling
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    clip_grad_norm: bool = True
    
    # Mixed precision training
    amp: bool = True
    amp_backend: str = "native"  # native, apex
    amp_level: str = "O1"
    
    # Exponential Moving Average
    ema: bool = True
    ema_decay: float = 0.9999
    ema_update_every: int = 10
    ema_start_step: int = 0
    
    def __post_init__(self):
        if isinstance(self.optimizer, str):
            self.optimizer = OptimizerType(self.optimizer)
        if isinstance(self.scheduler, str):
            self.scheduler = SchedulerType(self.scheduler)
        if isinstance(self.precision, str):
            self.precision = PrecisionType(self.precision)
            

@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""
    enabled: bool = True
    patience: int = 10
    min_delta: float = 0.001
    monitor: str = "val_loss"
    mode: str = "min"  # min, max
    restore_best_weights: bool = True
    

@dataclass
class LoggingConfig:
    """Logging configuration."""
    # Logger type
    logger: str = "tensorboard"  # tensorboard, wandb, none, all
    
    # TensorBoard settings
    tensorboard:
        log_dir: str = "./logs/tensorboard"
        flush_secs: int = 120
        max_queue: int = 20
        
    # Weights & Biases settings
    wandb:
        project: str = "cinematic-music-generation"
        entity: Optional[str] = None
        tags: List[str] = field(default_factory=lambda: ["music", "cinematic", "generation"])
        log_model: bool = True
        save_code: bool = True
        
    # General logging
    log_dir: str = "./logs"
    log_level: str = "INFO"
    log_interval: int = 100
    
    # Progress bar
    progress_bar: bool = True
    progress_refresh_rate: int = 50
    
    # Number of log images
    num_log_images: int = 4
    

@dataclass
class CheckpointConfig:
    """Checkpoint saving configuration."""
    save_dir: str = "./checkpoints"
    
    # Save settings
    save_interval: int = 1000
    save_top_k: int = 5
    save_last: bool = True
    
    # Checkpoint options
    save_weights_only: bool = False
    every_n_train_steps: Optional[int] = None
    every_n_epochs: Optional[int] = None
    
    # Resume training
    resume: bool = False
    resume_path: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    
    # Pretrained models
    pretrained_music_model: Optional[str] = None
    pretrained_voice_model: Optional[str] = None
    pretrained_lyrics_model: Optional[str] = None
    

@dataclass
class TrainingConfig:
    """Main training configuration."""
    # Project info
    name: str = "cinematic-music-generation"
    version: str = "1.0.0"
    description: str = "Large-scale deep learning for cinematic music generation"
    
    # Training loop
    epochs: int = 100
    max_steps: int = 1000000
    eval_interval: int = 1000
    val_check_interval: float = 1.0  # Can be int or float (fraction of epoch)
    limit_train_batches: Optional[int] = None
    limit_val_batches: Optional[int] = None
    num_sanity_val_steps: int = 0
    
    # Hardware
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    
    # Distributed training
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    
    # Optimization
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    
    # Early stopping
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    
    # Logging
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Checkpointing
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    # Reproducibility
    seed: int = 42
    
    # Debug settings
    debug: bool = False
    fast_dev_run: bool = False
    overfit_batches: float = 0.0
    
    # Data configuration reference (will be loaded separately)
    data_config: Optional[Dict] = None
    model_config: Optional[Dict] = None
    
    def __post_init__(self):
        # Validate configuration
        if self.epochs <= 0 and self.max_steps <= 0:
            raise ValueError("Either epochs or max_steps must be positive")
            
        if isinstance(self.hardware, dict):
            self.hardware = HardwareConfig(**self.hardware)
        if isinstance(self.distributed, dict):
            self.distributed = DistributedConfig(**self.distributed)
        if isinstance(self.optimization, dict):
            self.optimization = OptimizationConfig(**self.optimization)
        if isinstance(self.early_stopping, dict):
            self.early_stopping = EarlyStoppingConfig(**self.early_stopping)
        if isinstance(self.logging, dict):
            self.logging = LoggingConfig(**self.logging)
        if isinstance(self.checkpoint, dict):
            self.checkpoint = CheckpointConfig(**self.checkpoint)

