"""
Data cleaning utilities for cinematic music generation.
Handles bias reduction, quality filtering, and data validation.
"""

import numpy as np
import pandas as pd
import random
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class CleaningConfig:
    """Configuration for data cleaning."""
    # Duration filters
    min_duration: float = 5.0  # seconds
    max_duration: float = 600.0  # 10 minutes
    
    # Sample rate filter
    min_sample_rate: int = 22050
    target_sample_rate: int = 44100
    
    # Quality filters
    min_loudness: float = -60  # dB
    max_loudness: float = -3.0  # dB
    max_peak_normalized: float = 1.0
    
    # Silence filters
    max_silence_ratio: float = 0.5  # Max 50% silence
    min_voiced_ratio: float = 0.1  # Min 10% voiced content
    
    # Bias reduction
    balance_classes: bool = True
    max_samples_per_class: Optional[int] = None
    
    # Duplicate detection
    detect_duplicates: bool = True
    similarity_threshold: float = 0.95
    
    # Quality metrics to compute
    compute_quality_metrics: bool = True
    min_quality_score: float = 0.5


@dataclass
class SampleQuality:
    """Quality metrics for a single sample."""
    sample_id: str
    duration: float
    sample_rate: int
    loudness: float
    peak_level: float
    silence_ratio: float
    dynamic_range: float
    quality_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'sample_id': self.sample_id,
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'loudness': self.loudness,
            'peak_level': self.peak_level,
            'silence_ratio': self.silence_ratio,
            'dynamic_range': self.dynamic_range,
            'quality_score': self.quality_score,
            'issues': self.issues,
        }


class DataCleaner:
    """
    Comprehensive data cleaning for music datasets.
    Implements quality filtering, bias reduction, and validation.
    
    Args:
        config: Cleaning configuration
    """
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
        self.quality_cache = {}
    
    def clean_dataset(
        self,
        samples: List[Dict],
        return_stats: bool = True
    ) -> Tuple[List[Dict], Dict]:
        """
        Clean a list of samples.
        
        Args:
            samples: List of sample dictionaries
            return_stats: Whether to return cleaning statistics
            
        Returns:
            Tuple of (cleaned_samples, statistics)
        """
        stats = {
            'original_count': len(samples),
            'removed_count': 0,
            'removal_reasons': {},
            'class_distribution': {},
        }
        
        # Filter samples
        cleaned_samples = []
        
        for sample in samples:
            issues = self._check_sample(sample)
            
            if len(issues) == 0:
                cleaned_samples.append(sample)
            else:
                stats['removed_count'] += 1
                for issue in issues:
                    stats['removal_reasons'][issue] = stats['removal_reasons'].get(issue, 0) + 1
        
        # Apply bias reduction
        if self.config.balance_classes:
            cleaned_samples = self._balance_classes(cleaned_samples)
        
        # Remove duplicates
        if self.config.detect_duplicates:
            cleaned_samples = self._remove_duplicates(cleaned_samples)
        
        stats['final_count'] = len(cleaned_samples)
        stats['class_distribution'] = self._get_distribution(cleaned_samples)
        
        logger.info(f"Cleaned {stats['original_count']} samples -> {stats['final_count']} samples")
        logger.info(f"Removed {stats['removed_count']} samples")
        
        if return_stats:
            return cleaned_samples, stats
        return cleaned_samples
    
    def _check_sample(self, sample: Dict) -> List[str]:
        """Check a sample for issues."""
        issues = []
        
        # Check duration
        duration = sample.get('duration', 0)
        if duration < self.config.min_duration:
            issues.append('duration_too_short')
        elif duration > self.config.max_duration:
            issues.append('duration_too_long')
        
        # Check sample rate
        sample_rate = sample.get('sample_rate', 0)
        if sample_rate < self.config.min_sample_rate:
            issues.append('sample_rate_too_low')
        
        return issues
    
    def _balance_classes(self, samples: List[Dict]) -> List[Dict]:
        """Balance samples across classes."""
        # Group by class
        class_groups = {}
        for sample in samples:
            # Use emotion or genre as class label
            class_key = sample.get('emotion', sample.get('genre', 'unknown'))
            if class_key not in class_groups:
                class_groups[class_key] = []
            class_groups[class_key].append(sample)
        
        # Find target size
        min_count = min(len(group) for group in class_groups.values())
        
        if self.config.max_samples_per_class:
            min_count = min(min_count, self.config.max_samples_per_class)
        
        # Sample from each class
        balanced = []
        for class_name, group in class_groups.items():
            if len(group) > min_count:
                sampled = random.sample(group, min_count)
            else:
                sampled = group
            balanced.extend(sampled)
        
        logger.info(f"Balanced dataset: {len(balanced)} samples")
        return balanced
    
    def _remove_duplicates(self, samples: List[Dict]) -> List[Dict]:
        """Remove duplicate samples based on metadata."""
        seen_ids = set()
        unique_samples = []
        
        for sample in samples:
            sample_id = sample.get('id', sample.get('filepath', ''))
            if sample_id not in seen_ids:
                seen_ids.add(sample_id)
                unique_samples.append(sample)
        
        duplicates_removed = len(samples) - len(unique_samples)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicates")
        
        return unique_samples
    
    def _get_distribution(self, samples: List[Dict]) -> Dict:
        """Get class distribution."""
        distribution = {}
        for sample in samples:
            class_key = sample.get('emotion', sample.get('genre', 'unknown'))
            distribution[class_key] = distribution.get(class_key, 0) + 1
        return distribution


class BiasReducer:
    """
    Bias reduction for training data.
    Handles demographic, cultural, and stylistic biases.
    
    Args:
        config: Cleaning configuration
    """
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
    
    def reduce_bias(
        self,
        samples: List[Dict],
        bias_fields: List[str] = ['emotion', 'genre', 'tempo', 'key']
    ) -> List[Dict]:
        """
        Reduce bias in dataset by balancing across bias fields.
        
        Args:
            samples: List of samples
            bias_fields: Fields to consider for bias reduction
            
        Returns:
            Bias-reduced sample list
        """
        # Calculate current distribution
        original_dist = self._calculate_distribution(samples, bias_fields)
        
        # Identify over-represented groups
        max_allowed = self._calculate_max_allowed(samples, bias_fields)
        
        # Filter samples
        filtered = self._apply_distribution_limit(samples, bias_fields, max_allowed)
        
        # Report
        final_dist = self._calculate_distribution(filtered, bias_fields)
        
        logger.info("Bias reduction applied:")
        for field in bias_fields:
            logger.info(f"  {field}: {original_dist.get(field, {})} -> {final_dist.get(field, {})}")
        
        return filtered
    
    def _calculate_distribution(
        self,
        samples: List[Dict],
        bias_fields: List[str]
    ) -> Dict[str, Dict[str, int]]:
        """Calculate distribution across bias fields."""
        distribution = {}
        
        for field in bias_fields:
            field_dist = {}
            for sample in samples:
                value = str(sample.get(field, 'unknown'))
                field_dist[value] = field_dist.get(value, 0) + 1
            distribution[field] = field_dist
        
        return distribution
    
    def _calculate_max_allowed(
        self,
        samples: List[Dict],
        bias_fields: List[str]
    ) -> Dict[str, Dict[str, int]]:
        """Calculate maximum allowed samples per group."""
        total = len(samples)
        
        # Target maximum percentage per group
        max_pct = 0.5  # No group should be more than 50%
        
        # Find most common value for each field
        max_allowed = {}
        for field in bias_fields:
            field_dist = {}
            for sample in samples:
                value = str(sample.get(field, 'unknown'))
                field_dist[value] = field_dist.get(value, 0) + 1
            
            max_count = max(field_dist.values())
            if max_count / total > max_pct:
                # Need to reduce
                allowed = int(total * max_pct / len(field_dist))
                for value in field_dist:
                    field_dist[value] = min(field_dist[value], allowed)
                max_allowed[field] = field_dist
            else:
                max_allowed[field] = None
        
        return max_allowed
    
    def _apply_distribution_limit(
        self,
        samples: List[Dict],
        bias_fields: List[str],
        max_allowed: Dict
    ) -> List[Dict]:
        """Apply distribution limits."""
        counts = {field: {} for field in bias_fields}
        filtered = []
        
        for sample in samples:
            include = True
            
            for field in bias_fields:
                value = str(sample.get(field, 'unknown'))
                
                if max_allowed.get(field) and value in max_allowed[field]:
                    current = counts[field].get(value, 0)
                    limit = max_allowed[field][value]
                    
                    if current >= limit:
                        include = False
                        break
                    
                    counts[field][value] = current + 1
            
            if include:
                filtered.append(sample)
        
        return filtered


class QualityAssessor:
    """
    Assess audio quality metrics.
    Computes various quality indicators for samples.
    """
    
    def __init__(self):
        pass
    
    def assess(self, audio: np.ndarray, sample_rate: int) -> SampleQuality:
        """
        Assess quality of an audio sample.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            
        Returns:
            SampleQuality object with metrics
        """
        try:
            import librosa
            
            # Basic metrics
            duration = len(audio) / sample_rate
            
            # Calculate loudness (RMS)
            rms = np.sqrt(np.mean(audio**2))
            loudness = 20 * np.log10(rms + 1e-10) if rms > 0 else -60
            
            # Peak level
            peak = np.max(np.abs(audio))
            peak_db = 20 * np.log10(peak + 1e-10) if peak > 0 else -60
            
            # Silence ratio
            threshold = 10 ** (-40 / 20)  # -40 dB
            non_silent = np.sum(np.abs(audio) > threshold)
            silence_ratio = 1 - (non_silent / len(audio)) if len(audio) > 0 else 1
            
            # Dynamic range
            max_db = 20 * np.log10(np.max(audio + 1e-10))
            min_db = 20 * np.log10(np.min(np.abs(audio) + 1e-10))
            dynamic_range = max_db - min_db
            
            # Calculate quality score (simplified)
            quality_score = self._calculate_quality_score(
                duration, loudness, peak_db, silence_ratio, dynamic_range
            )
            
            # Identify issues
            issues = []
            if duration < 5.0:
                issues.append('too_short')
            if loudness < -50:
                issues.append('too_quiet')
            if peak_db > 0:
                issues.append('clipping')
            if silence_ratio > 0.5:
                issues.append('too_much_silence')
            
            return SampleQuality(
                sample_id='',
                duration=duration,
                sample_rate=sample_rate,
                loudness=loudness,
                peak_level=peak_db,
                silence_ratio=silence_ratio,
                dynamic_range=dynamic_range,
                quality_score=quality_score,
                issues=issues
            )
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return SampleQuality(
                sample_id='',
                duration=0,
                sample_rate=sample_rate,
                loudness=-60,
                peak_level=-60,
                silence_ratio=1.0,
                dynamic_range=0,
                quality_score=0,
                issues=['assessment_failed']
            )
    
    def _calculate_quality_score(
        self,
        duration: float,
        loudness: float,
        peak_db: float,
        silence_ratio: float,
        dynamic_range: float
    ) -> float:
        """Calculate overall quality score (0-1)."""
        score = 0.0
        
        # Duration score (prefer 10s - 5min)
        if 10 <= duration <= 300:
            score += 0.3
        elif 5 <= duration <= 600:
            score += 0.2
        else:
            score += 0.1
        
        # Loudness score (prefer -20 to -6 dB)
        if -20 <= loudness <= -6:
            score += 0.3
        elif -40 <= loudness <= -3:
            score += 0.2
        else:
            score += 0.1
        
        # Peak score (avoid clipping)
        if peak_db < -0.5:
            score += 0.2
        else:
            score += 0.0
        
        # Silence score
        if silence_ratio < 0.2:
            score += 0.2
        elif silence_ratio < 0.5:
            score += 0.1
        
        return min(score, 1.0)


class MetadataValidator:
    """
    Validate and clean sample metadata.
    Ensures consistency and correctness of metadata.
    """
    
    VALID_EMOTIONS = [
        'neutral', 'happy', 'sad', 'angry', 'fearful', 
        'surprised', 'disgusted', 'hopeful', 'tense', 'calm'
    ]
    
    VALID_GENRES = [
        'cinematic', 'orchestral', 'electronic', 'ambient',
        'action', 'drama', 'horror', 'comedy', 'romance', 'documentary'
    ]
    
    KEY_SIGNATURES = [
        'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
        'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm'
    ]
    
    def __init__(self):
        pass
    
    def validate(self, sample: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a sample's metadata.
        
        Args:
            sample: Sample dictionary
            
        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        
        # Check required fields
        if 'id' not in sample:
            errors.append('missing_id')
        
        if 'duration' in sample:
            if sample['duration'] <= 0:
                errors.append('invalid_duration')
        
        # Validate emotion
        if 'emotion' in sample:
            if sample['emotion'] not in self.VALID_EMOTIONS:
                errors.append(f'invalid_emotion: {sample["emotion"]}')
        
        # Validate genre
        if 'genre' in sample:
            if sample['genre'] not in self.VALID_GENRES:
                errors.append(f'invalid_genre: {sample["genre"]}')
        
        # Validate key
        if 'key' in sample:
            if sample['key'] not in self.KEY_SIGNATURES:
                errors.append(f'invalid_key: {sample["key"]}')
        
        # Validate tempo
        if 'tempo' in sample:
            if not (30 <= sample['tempo'] <= 300):
                errors.append('invalid_tempo')
        
        return len(errors) == 0, errors
    
    def clean(self, sample: Dict) -> Dict:
        """
        Clean and normalize sample metadata.
        
        Args:
            sample: Sample dictionary
            
        Returns:
            Cleaned sample dictionary
        """
        cleaned = sample.copy()
        
        # Clean emotion
        if 'emotion' in cleaned:
            emotion = cleaned['emotion'].lower().strip()
            if emotion not in self.VALID_EMOTIONS:
                # Try to find closest match
                emotion = self._find_closest(emotion, self.VALID_EMOTIONS)
            cleaned['emotion'] = emotion
        
        # Clean genre
        if 'genre' in cleaned:
            genre = cleaned['genre'].lower().strip()
            if genre not in self.VALID_GENRES:
                genre = self._find_closest(genre, self.VALID_GENRES)
            cleaned['genre'] = genre
        
        # Clean key
        if 'key' in cleaned:
            key = cleaned['key'].replace(' ', '').replace('-', '')
            if key not in self.KEY_SIGNATURES:
                key = 'C'  # Default
            cleaned['key'] = key
        
        # Ensure tempo is numeric
        if 'tempo' in cleaned:
            try:
                cleaned['tempo'] = float(cleaned['tempo'])
                cleaned['tempo'] = max(30, min(300, cleaned['tempo']))
            except (ValueError, TypeError):
                cleaned['tempo'] = 120.0
        
        return cleaned
    
    def _find_closest(self, value: str, options: List[str]) -> str:
        """Find closest matching option using simple string matching."""
        value = value.lower()
        
        for option in options:
            if value in option.lower() or option.lower() in value:
                return option
        
        return options[0]  # Return first option as default


def create_cleaning_pipeline(
    config: Optional[CleaningConfig] = None,
    bias_fields: List[str] = ['emotion', 'genre']
) -> Tuple[DataCleaner, BiasReducer, QualityAssessor, MetadataValidator]:
    """
    Create a complete cleaning pipeline.
    
    Args:
        config: Cleaning configuration
        bias_fields: Fields for bias reduction
        
    Returns:
        Tuple of cleaning components
    """
    config = config or CleaningConfig()
    
    cleaner = DataCleaner(config)
    reducer = BiasReducer(config)
    assessor = QualityAssessor()
    validator = MetadataValidator()
    
    return cleaner, reducer, assessor, validator


def clean_and_validate_dataset(
    samples: List[Dict],
    config: Optional[CleaningConfig] = None,
    bias_fields: List[str] = ['emotion', 'genre']
) -> Tuple[List[Dict], Dict]:
    """
    Complete dataset cleaning and validation.
    
    Args:
        samples: List of samples
        config: Cleaning configuration
        bias_fields: Fields for bias reduction
        
    Returns:
        Tuple of (cleaned_samples, statistics)
    """
    cleaner, reducer, assessor, validator = create_cleaning_pipeline(config, bias_fields)
    
    # Validate and clean metadata
    validated = []
    for sample in samples:
        is_valid, errors = validator.validate(sample)
        if is_valid:
            cleaned_sample = validator.clean(sample)
            validated.append(cleaned_sample)
        else:
            logger.debug(f"Sample validation failed: {errors}")
    
    # Apply bias reduction
    balanced = reducer.reduce_bias(validated, bias_fields)
    
    # Final quality cleaning
    cleaned, stats = cleaner.clean_dataset(balanced)
    
    return cleaned, stats

