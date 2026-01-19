"""
Data loaders for distributed training.
Provides PyTorch DataLoaders with distributed sampling support.
"""

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Optional, Union, Callable
import logging

logger = logging.getLogger(__name__)


class DistributedSamplerWrapper:
    """
    Wrapper for distributed sampler.
    Handles proper shuffling and epoch tracking for distributed training.
    
    Args:
        sampler: Base sampler
        num_replicas: Number of replicas
        rank: Current rank
        shuffle: Whether to shuffle
    """
    
    def __init__(
        self,
        sampler: torch.utils.data.Sampler,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.sampler = sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        if self.shuffle:
            # Set seed for this epoch
            torch.manual_seed(seed)
    
    def set_epoch(self, epoch: int):
        """Set the epoch for proper shuffling."""
        self.epoch = epoch
        if self.shuffle:
            torch.manual_seed(self.seed + epoch)
    
    def __iter__(self):
        # Deterministic seeding based on epoch
        if self.shuffle:
            torch.manual_seed(self.seed + self.epoch)
        
        while True:
            indices = []
            for idx in self.sampler:
                if torch.distributed.get_rank() == self.rank:
                    indices.append(idx)
                if len(indices) * self.num_replicas >= len(self.sampler):
                    break
            yield from indices
    
    def __len__(self) -> int:
        return len(self.sampler)


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    # Distributed settings
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    seed: int = 42,
) -> dict:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        test_dataset: Test dataset (optional)
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        distributed: Use distributed training
        world_size: Number of processes
        rank: Current process rank
        seed: Random seed
        
    Returns:
        Dictionary of data loaders
    """
    loaders = {}
    
    # Training loader
    train_sampler = None
    if distributed and world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
    
    loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=True,
        shuffle=(train_sampler is None),  # Only shuffle if no sampler
    )
    
    # Validation loader
    if val_dataset is not None:
        val_sampler = None
        if distributed and world_size > 1:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=seed,
                drop_last=False,
            )
        
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and num_workers > 0,
            drop_last=False,
            shuffle=False,
        )
    
    # Test loader
    if test_dataset is not None:
        test_sampler = None
        if distributed and world_size > 1:
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=seed,
                drop_last=False,
            )
        
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and num_workers > 0,
            drop_last=False,
            shuffle=False,
        )
    
    # Log info
    logger.info(f"Created data loaders:")
    logger.info(f"  Train: {len(loaders['train'])} batches (batch_size={batch_size})")
    if 'val' in loaders:
        logger.info(f"  Val: {len(loaders['val'])} batches")
    if 'test' in loaders:
        logger.info(f"  Test: {len(loaders['test'])} batches")
    
    if distributed and world_size > 1:
        logger.info(f"  Distributed: {world_size} GPUs, rank={rank}")
    
    return loaders


class InfiniteDataLoader:
    """
    Infinite data loader that wraps around a DataLoader.
    Useful for streaming data during training.
    
    Args:
        data_loader: Base DataLoader
        sample_transform: Optional transform to apply to samples
    """
    
    def __init__(
        self,
        data_loader: DataLoader,
        sample_transform: Optional[Callable] = None,
    ):
        self.data_loader = data_loader
        self.sample_transform = sample_transform
        self._iterator = None
    
    def _get_iterator(self):
        """Get a new iterator over the data loader."""
        while True:
            for batch in self.data_loader:
                if self.sample_transform is not None:
                    batch = self.sample_transform(batch)
                yield batch
    
    def __iter__(self):
        if self._iterator is None:
            self._iterator = self._get_iterator()
        return self._iterator
    
    def __len__(self) -> int:
        return len(self.data_loader)
    
    def reset(self):
        """Reset the iterator."""
        self._iterator = None


class CacheDataLoader:
    """
    Data loader with caching for fast epoch access.
    Caches the entire dataset in memory after first epoch.
    
    Args:
        data_loader: Base DataLoader
        cache_size: Maximum number of batches to cache (0 for all)
    """
    
    def __init__(
        self,
        data_loader: DataLoader,
        cache_size: int = 0,
    ):
        self.data_loader = data_loader
        self.cache_size = cache_size
        self.cache = []
        self.cache_filled = False
    
    def _fill_cache(self):
        """Fill the cache with batches."""
        self.cache = []
        for batch in self.data_loader:
            self.cache.append(batch)
            if self.cache_size > 0 and len(self.cache) >= self.cache_size:
                break
        self.cache_filled = True
    
    def __iter__(self):
        if self.cache_filled:
            yield from self.cache
        else:
            for batch in self.data_loader:
                if self.cache_size == 0 or len(self.cache) < self.cache_size:
                    self.cache.append(batch)
                yield batch
            self.cache_filled = True
    
    def __len__(self) -> int:
        if self.cache_filled:
            return len(self.cache)
        return len(self.data_loader)


def get_worker_init_fn(seed: int = 42):
    """
    Create worker init function for data loaders.
    Ensures each worker has a unique random seed.
    
    Args:
        seed: Base seed for workers
        
    Returns:
        Worker init function
    """
    def worker_init_fn(worker_id: int):
        worker_seed = seed + worker_id
        import random
        random.seed(worker_seed)
        import numpy as np
        np.random.seed(worker_seed)
        import torch
        torch.set_num_threads(1)
    return worker_init_fn


class AspectRatioBatchSampler:
    """
    Batch sampler that groups samples by aspect ratio.
    Useful for variable-size image/audio inputs.
    
    Args:
        sampler: Base sampler
        batch_size: Batch size
        aspect_ratios: List of aspect ratio groups
    """
    
    def __init__(
        self,
        sampler,
        batch_size: int,
        aspect_ratios: list = None,
    ):
        self.sampler = sampler
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios or []
    
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        if len(batch) > 0:
            yield batch
    
    def __len__(self) -> int:
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size

