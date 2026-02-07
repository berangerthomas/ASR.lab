"""
Metrics computation module with configurable text normalization.

This module computes WER, CER, and other metrics from raw transcriptions
with user-specified normalization options. Each normalization preset
is treated as a separate dimension in the grid search.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import jiwer

logger = logging.getLogger(__name__)


# Text normalization presets - simple binary: raw or normalized
# These are applied as a grid search dimension, like degradation or enhancement
TEXT_NORM_PRESETS = {
    'raw': {
        'display_name': 'Brut',
        'lowercase': False,
        'remove_punctuation': False,
        'normalize_spaces': False,
        'expand_contractions': False,
    },
    'normalized': {
        'display_name': 'Normalisé',
        'lowercase': True,
        'remove_punctuation': True,
        'normalize_spaces': True,
        'expand_contractions': False,
    },
}


@dataclass
class NormalizationOptions:
    """Text normalization options for metric computation."""
    lowercase: bool = True
    remove_punctuation: bool = True
    normalize_spaces: bool = True  # RemoveMultipleSpaces + Strip
    expand_contractions: bool = False  # English only
    
    def to_dict(self) -> Dict[str, bool]:
        """Convert to dictionary."""
        return {
            'lowercase': self.lowercase,
            'remove_punctuation': self.remove_punctuation,
            'normalize_spaces': self.normalize_spaces,
            'expand_contractions': self.expand_contractions,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'NormalizationOptions':
        """Create from dictionary."""
        return cls(
            lowercase=d.get('lowercase', True),
            remove_punctuation=d.get('remove_punctuation', True),
            normalize_spaces=d.get('normalize_spaces', True),
            expand_contractions=d.get('expand_contractions', False),
        )
    
    @classmethod
    def from_preset(cls, preset_name: str) -> 'NormalizationOptions':
        """Create from a preset name."""
        if preset_name not in TEXT_NORM_PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(TEXT_NORM_PRESETS.keys())}")
        preset = TEXT_NORM_PRESETS[preset_name]
        return cls(
            lowercase=preset['lowercase'],
            remove_punctuation=preset['remove_punctuation'],
            normalize_spaces=preset['normalize_spaces'],
            expand_contractions=preset['expand_contractions'],
        )


def normalize_text(text: str, options: NormalizationOptions) -> str:
    """
    Apply text normalization transforms and return the normalized text.
    
    This allows displaying the normalized text in the diff view.
    
    Args:
        text: Input text
        options: Normalization options
        
    Returns:
        Normalized text string
    """
    result = text
    
    if options.expand_contractions:
        # Apply English contraction expansion
        contractions = {
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "won't": "will not", "wouldn't": "would not", "couldn't": "could not",
            "shouldn't": "should not", "can't": "cannot", "wasn't": "was not",
            "weren't": "were not", "isn't": "is not", "aren't": "are not",
            "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
            "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
            "it's": "it is", "we're": "we are", "they're": "they are",
            "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
            "i'll": "i will", "you'll": "you will", "he'll": "he will", "she'll": "she will",
            "we'll": "we will", "they'll": "they will", "i'd": "i would", "you'd": "you would",
            "he'd": "he would", "she'd": "she would", "we'd": "we would", "they'd": "they would",
            "let's": "let us", "that's": "that is", "who's": "who is", "what's": "what is",
            "here's": "here is", "there's": "there is", "where's": "where is",
        }
        # Case-insensitive replacement
        for contraction, expansion in contractions.items():
            pattern = re.compile(re.escape(contraction), re.IGNORECASE)
            result = pattern.sub(expansion, result)
    
    if options.lowercase:
        result = result.lower()
    
    if options.remove_punctuation:
        result = re.sub(r'[^\w\s]', '', result)
    
    if options.normalize_spaces:
        result = re.sub(r'\s+', ' ', result).strip()
    
    return result


def create_word_transforms(options: NormalizationOptions) -> jiwer.Compose:
    """
    Create jiwer transformation pipeline for WER computation.
    
    Args:
        options: Normalization options
        
    Returns:
        jiwer.Compose transform pipeline
    """
    transforms: List[jiwer.AbstractTransform] = []
    
    if options.lowercase:
        transforms.append(jiwer.ToLowerCase())
    
    if options.remove_punctuation:
        transforms.append(jiwer.RemovePunctuation())
    
    if options.expand_contractions:
        transforms.append(jiwer.ExpandCommonEnglishContractions())
    
    if options.normalize_spaces:
        transforms.append(jiwer.RemoveMultipleSpaces())
        transforms.append(jiwer.Strip())
    
    # Required by jiwer for WER
    transforms.append(jiwer.ReduceToListOfListOfWords())
    
    return jiwer.Compose(transforms)


def create_char_transforms(options: NormalizationOptions) -> jiwer.Compose:
    """
    Create jiwer transformation pipeline for CER computation.
    
    Args:
        options: Normalization options
        
    Returns:
        jiwer.Compose transform pipeline
    """
    transforms: List[jiwer.AbstractTransform] = []
    
    if options.lowercase:
        transforms.append(jiwer.ToLowerCase())
    
    if options.remove_punctuation:
        transforms.append(jiwer.RemovePunctuation())
    
    if options.expand_contractions:
        transforms.append(jiwer.ExpandCommonEnglishContractions())
    
    if options.normalize_spaces:
        transforms.append(jiwer.RemoveMultipleSpaces())
        transforms.append(jiwer.Strip())
    
    # Required by jiwer for CER
    transforms.append(jiwer.ReduceToListOfListOfChars())
    
    return jiwer.Compose(transforms)


@dataclass
class MetricsResult:
    """Result of metrics computation for a single transcription."""
    wer: Optional[float] = None
    cer: Optional[float] = None
    mer: Optional[float] = None
    wil: Optional[float] = None
    wip: Optional[float] = None


class MetricsComputer:
    """
    Computes ASR metrics with configurable text normalization.
    
    This class is used by the report generator to calculate metrics
    on-demand from raw transcriptions stored in the database.
    """
    
    def __init__(self, options: Optional[NormalizationOptions] = None):
        """
        Initialize the metrics computer.
        
        Args:
            options: Text normalization options. Uses defaults if None.
        """
        self.options = options or NormalizationOptions()
        self.word_transforms = create_word_transforms(self.options)
        self.char_transforms = create_char_transforms(self.options)
    
    def compute(self, hypothesis: str, reference: str) -> MetricsResult:
        """
        Compute all metrics for a single transcription pair.
        
        Args:
            hypothesis: ASR output text
            reference: Ground truth text
            
        Returns:
            MetricsResult with all computed metrics
        """
        result = MetricsResult()
        
        try:
            # WER
            result.wer = jiwer.wer(
                reference=reference,
                hypothesis=hypothesis,
                reference_transform=self.word_transforms,
                hypothesis_transform=self.word_transforms,
            )
        except Exception as e:
            logger.warning(f"WER computation failed: {e}")
        
        try:
            # CER
            result.cer = jiwer.cer(
                reference=reference,
                hypothesis=hypothesis,
                reference_transform=self.char_transforms,
                hypothesis_transform=self.char_transforms,
            )
        except Exception as e:
            logger.warning(f"CER computation failed: {e}")
        
        try:
            # MER (Match Error Rate)
            result.mer = jiwer.mer(
                reference=reference,
                hypothesis=hypothesis,
                reference_transform=self.word_transforms,
                hypothesis_transform=self.word_transforms,
            )
        except Exception as e:
            logger.warning(f"MER computation failed: {e}")
        
        try:
            # WIL (Word Information Lost)
            result.wil = jiwer.wil(
                reference=reference,
                hypothesis=hypothesis,
                reference_transform=self.word_transforms,
                hypothesis_transform=self.word_transforms,
            )
        except Exception as e:
            logger.warning(f"WIL computation failed: {e}")
        
        try:
            # WIP (Word Information Preserved)
            result.wip = jiwer.wip(
                reference=reference,
                hypothesis=hypothesis,
                reference_transform=self.word_transforms,
                hypothesis_transform=self.word_transforms,
            )
        except Exception as e:
            logger.warning(f"WIP computation failed: {e}")
        
        return result


def compute_with_text_norm_presets(
    hypothesis: str,
    reference: str,
    presets: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Compute metrics for each text normalization preset.
    
    Each preset becomes a separate result, enabling grid search
    over text normalization options. The normalized texts are included
    for display in the diff view.
    
    Args:
        hypothesis: ASR output text (raw)
        reference: Ground truth text (raw)
        presets: List of preset names to use. Defaults to all presets.
        
    Returns:
        List of dicts, one per preset, with:
        - text_norm: preset name
        - text_norm_display: human-readable preset name
        - reference_normalized: normalized reference text
        - hypothesis_normalized: normalized hypothesis text
        - metrics: dict with wer, cer, mer, wil, wip
    """
    if presets is None:
        presets = list(TEXT_NORM_PRESETS.keys())
    
    results = []
    
    for preset_name in presets:
        if preset_name not in TEXT_NORM_PRESETS:
            logger.warning(f"Unknown preset '{preset_name}', skipping")
            continue
        
        preset = TEXT_NORM_PRESETS[preset_name]
        options = NormalizationOptions.from_preset(preset_name)
        
        # Normalize texts for display
        ref_normalized = normalize_text(reference, options)
        hyp_normalized = normalize_text(hypothesis, options)
        
        # Compute metrics
        computer = MetricsComputer(options)
        metrics = computer.compute(hypothesis, reference)
        
        results.append({
            'text_norm': preset_name,
            'text_norm_display': preset['display_name'],
            'reference_normalized': ref_normalized,
            'hypothesis_normalized': hyp_normalized,
            'metrics': {
                'wer': metrics.wer,
                'cer': metrics.cer,
                'mer': metrics.mer,
                'wil': metrics.wil,
                'wip': metrics.wip,
            }
        })
    
    return results
