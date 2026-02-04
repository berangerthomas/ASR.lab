import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ..audio.degradation import AudioDegradation
from ..audio.processor import AudioProcessor
from ..audio.enhancement import DemucsEnhancer, DeepFilterNetEnhancer, AudioEnhancer
from ..audio.normalization import AudioNormalizer
from ..config.benchmark_config import ConfigLoader

logger = logging.getLogger(__name__)

class DataManager:
    """
    Manages the preparation and loading of datasets for benchmarking.
    Handles audio processing, degradation, and enhancement.
    """

    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
        self.audio_processor = self._initialize_audio_processor()
        self.audio_degradation = self._initialize_audio_degradation()
        self.audio_enhancers = self._initialize_audio_enhancers()
        self.audio_normalizer = AudioNormalizer()
        
    def prepare_datasets(self) -> List[Dict[str, Any]]:
        """
        Prepares the audio datasets for benchmarking, including degradations and enhancements.
        Returns a list of dataset specifications.
        """
        logger.info("Preparing datasets...")
        datasets = []
        
        # Get data paths from config
        config = self.config_loader.get_config()
        audio_source_dir = Path(config.data.audio_source_dir)
        reference_dir = Path(config.data.reference_dir)
        processed_dir = Path(config.data.processed_dir)
        processed_dir.mkdir(exist_ok=True, parents=True)

        # Collect all tasks to be done
        tasks = []
        
        # Find all audio files in the source directory
        # Check for manifest file first
        manifest_path = audio_source_dir / "manifest.json"
        audio_files = []
        
        if manifest_path.exists():
            logger.info(f"Loading dataset from manifest: {manifest_path}")
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest_data = json.load(f)
                    # Expecting list of dicts: [{"audio_filepath": "...", "text": "...", "lang": "..."}]
                    for item in manifest_data:
                        audio_path = Path(item.get("audio_filepath"))
                        if not audio_path.is_absolute():
                            audio_path = audio_source_dir / audio_path
                        
                        if audio_path.exists():
                            audio_files.append({
                                "path": audio_path,
                                "text": item.get("text"),
                                "lang": item.get("lang")
                            })
                        else:
                            logger.warning(f"Audio file in manifest not found: {audio_path}")
            except Exception as e:
                logger.error(f"Failed to load manifest: {e}")
        
        # Fallback to glob if no manifest or empty
        if not audio_files:
            logger.info("No manifest found or loaded. Falling back to file globbing.")
            for f in audio_source_dir.glob("*.wav"):
                audio_files.append({"path": f})
        
        if not audio_files:
            logger.warning(f"No audio files found in {audio_source_dir}")
            return []

        for item in audio_files:
            source_audio_path = item["path"]
            
            # Determine language and reference text
            if "lang" in item:
                lang_code = item["lang"]
            else:
                # Fallback to filename convention
                try:
                    lang_code = source_audio_path.stem.split('_')[0]
                except IndexError:
                    logger.warning(f"Skipping file with invalid naming convention: {source_audio_path.name}")
                    continue
            
            if "text" in item:
                reference_text = item["text"]
            else:
                # Fallback to text file
                reference_text_path = reference_dir / f"{source_audio_path.stem}.txt"
                if not reference_text_path.is_file():
                    logger.warning(f"Reference text not found for audio file: {source_audio_path}")
                    continue
                try:
                    with open(reference_text_path, 'r', encoding='utf-8') as f:
                        reference_text = f.read().strip()
                except Exception as e:
                    logger.error(f"Could not read reference text at {reference_text_path}: {e}")
                    continue

            # Add original task
            tasks.append({
                "type": "original",
                "source_path": source_audio_path,
                "processed_dir": processed_dir,
                "lang_code": lang_code,
                "reference_text": reference_text
            })

            # Add degradation tasks
            if self.audio_degradation:
                config = self.config_loader.get_config()
                degradations = config.degradations.presets if config.degradations else []
                for preset in degradations:
                    tasks.append({
                        "type": "degradation",
                        "source_path": source_audio_path,
                        "processed_dir": processed_dir,
                        "lang_code": lang_code,
                        "reference_text": reference_text,
                        "preset": preset.model_dump()
                    })

        # Add enhancement tasks
        if self.audio_enhancers:
            enhancement_tasks = []
            for task in tasks:
                # We can enhance original or degraded files.
                # But we don't have the output path of the task yet because it's not processed.
                # We need to predict the output path or restructure.
                
                # Let's predict the output path based on task type
                if task["type"] == "original":
                    input_path = processed_dir / f"{task['source_path'].stem}_original.wav"
                    base_name = f"{task['source_path'].stem}_original"
                elif task["type"] == "degradation":
                    input_path = processed_dir / f"{task['source_path'].stem}_degraded_{task['preset']['name']}.wav"
                    base_name = f"{task['source_path'].stem}_degraded_{task['preset']['name']}"
                else:
                    continue
                
                for enhancer_name, enhancer in self.audio_enhancers.items():
                    enhancement_tasks.append({
                        "type": "enhancement",
                        "source_path": task["source_path"], # Keep original source for reference
                        "input_path": input_path, # The file to enhance (will be generated by previous task)
                        "processed_dir": processed_dir,
                        "lang_code": task["lang_code"],
                        "reference_text": task["reference_text"],
                        "base_name": base_name,
                        "enhancer_name": enhancer_name
                    })
            tasks.extend(enhancement_tasks)

        # Process audio files (potentially in parallel if we wanted, but audio processing is fast enough usually)
        # We will just loop for now to ensure file integrity
        logger.info(f"Processing {len(tasks)} audio preparation tasks...")
        
        intermediate_datasets = []
        for task in tqdm(tasks, desc="Preparing Audio (Base)"):
            try:
                dataset_info = self._process_single_task(task)
                intermediate_datasets.append(dataset_info)
            except Exception as e:
                logger.error(f"Failed to process task for {task['source_path']}: {e}")

        # Apply Normalization Grid Search
        final_datasets = []
        normalizations = config.normalizations
        
        # If no normalizations defined, treat as "None" (backward compatibility)
        if not normalizations:
             for ds in intermediate_datasets:
                 ds["normalization"] = "None"
                 final_datasets.append(ds)
        else:
            logger.info(f"Applying {len(normalizations)} normalization configurations...")
            for ds in tqdm(intermediate_datasets, desc="Normalizing Audio"):
                for norm_config in normalizations:
                    if not norm_config.enabled:
                        continue
                        
                    if norm_config.method == "none":
                        # Still prepare for ASR (ensure mono, 16kHz, PCM)
                        new_stem = f"{ds['audio_path'].stem}_{norm_config.name}"
                        new_path = processed_dir / f"{new_stem}.wav"
                        
                        if not new_path.exists():
                            try:
                                self.audio_normalizer.prepare_for_asr(
                                    ds['audio_path'], 
                                    new_path
                                )
                            except Exception as e:
                                logger.error(f"ASR preparation failed for {ds['name']}: {e}")
                                continue
                        
                        new_ds = ds.copy()
                        new_ds["normalization"] = norm_config.name
                        new_ds["name"] = f"{ds['name']}_{norm_config.name}"
                        new_ds["audio_path"] = new_path
                        final_datasets.append(new_ds)
                        continue

                    # Construct new path
                    new_stem = f"{ds['audio_path'].stem}_{norm_config.name}"
                    new_path = processed_dir / f"{new_stem}.wav"
                    
                    if not new_path.exists():
                        try:
                            self.audio_normalizer.normalize(
                                ds['audio_path'], 
                                new_path, 
                                target_loudness=norm_config.target_loudness
                            )
                        except Exception as e:
                            logger.error(f"Normalization failed for {ds['name']}: {e}")
                            continue
                    
                    new_ds = ds.copy()
                    new_ds["name"] = f"{ds['name']}_{norm_config.name}"
                    new_ds["audio_path"] = new_path
                    new_ds["normalization"] = norm_config.name
                    final_datasets.append(new_ds)

        logger.info(f"Prepared {len(final_datasets)} final datasets.")
        return final_datasets

    def _process_single_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single audio preparation task."""
        source_path = task["source_path"]
        processed_dir = task["processed_dir"]
        
        if task["type"] == "original":
            processed_audio_path = processed_dir / f"{source_path.stem}_original.wav"
            # Only process if not exists or force overwrite (could add config for this)
            if not processed_audio_path.exists():
                self.audio_processor.process_audio(source_path, processed_audio_path)
            
            return {
                "name": f"{source_path.stem}_original",
                "language": task["lang_code"],
                "audio_path": processed_audio_path,
                "reference_text": task["reference_text"],
                "degradation": "None",
                "enhancement": "None"
            }
            
        elif task["type"] == "degradation":
            preset = task["preset"]
            # We need the original processed file as base? Or source? 
            # Usually degradations apply to the clean source.
            # But to be consistent with normalization, we might want to normalize first.
            # For simplicity, let's assume degradation handles source -> degraded.
            
            # First ensure we have a normalized base if needed, but let's stick to direct source for now
            # or use the already processed original if available.
            # Let's use the source directly to avoid dependency chains here.
            
            # Ideally: Source -> Normalize -> Degrade
            # Current flow: Source -> Degrade
            
            degraded_path = processed_dir / f"{source_path.stem}_degraded_{preset['name']}.wav"
            
            # Optimization: Check if exists
            if not degraded_path.exists():
                # We might want to normalize first. 
                # Let's create a temp normalized path if we want strict pipeline, 
                # but for now let's trust the degradation module or input.
                # Actually, let's use the _original.wav as input if it exists to ensure normalization consistency
                original_processed_path = processed_dir / f"{source_path.stem}_original.wav"
                input_for_degradation = original_processed_path if original_processed_path.exists() else source_path
                
                self.audio_degradation.apply_degradation(
                    input_for_degradation,
                    degraded_path,
                    preset["preset_name"]
                )
            
            return {
                "name": f"{source_path.stem}_degraded_{preset['name']}",
                "language": task["lang_code"],
                "audio_path": degraded_path,
                "reference_text": task["reference_text"],
                "degradation": preset['name'],
                "enhancement": "None"
            }

        elif task["type"] == "enhancement":
            input_path = task["input_path"]
            base_name = task["base_name"]
            enhancer_name = task["enhancer_name"]
            
            # Ensure input exists (it should have been created by previous tasks)
            if not input_path.exists():
                # If it doesn't exist, we might need to trigger its creation or skip
                # Since we iterate sequentially and append tasks, the previous tasks should have run.
                # But if they failed or were skipped, we have an issue.
                # For now, let's try to use source if input missing (fallback)
                logger.warning(f"Input for enhancement not found: {input_path}. Skipping.")
                raise FileNotFoundError(f"Input file {input_path} not found.")

            enhanced_path = processed_dir / f"{base_name}_enhanced_{enhancer_name}.wav"
            
            if not enhanced_path.exists():
                enhancer = self.audio_enhancers.get(enhancer_name)
                if enhancer:
                    enhancer.enhance(input_path, enhanced_path)
                else:
                    logger.error(f"Enhancer {enhancer_name} not found.")
                    raise ValueError(f"Enhancer {enhancer_name} not found.")
            
            # Determine degradation from base_name or task info
            degradation = "None"
            if "_degraded_" in base_name:
                # base_name format: {stem}_degraded_{preset_name}
                parts = base_name.split("_degraded_")
                if len(parts) > 1:
                    degradation = parts[1]
            
            return {
                "name": f"{base_name}_enhanced_{enhancer_name}",
                "language": task["lang_code"],
                "audio_path": enhanced_path,
                "reference_text": task["reference_text"],
                "degradation": degradation,
                "enhancement": enhancer_name
            }
        
        raise ValueError(f"Unknown task type: {task['type']}")

    def _initialize_audio_processor(self) -> AudioProcessor:
        """Initializes the audio processor."""
        # With Pydantic, we pass the dict representation or update AudioProcessor to accept the model
        config = self.config_loader.get_config()
        return AudioProcessor(config.audio_processing.model_dump())

    def _initialize_audio_degradation(self) -> Optional[AudioDegradation]:
        """Initializes the audio degradation module if configured."""
        config = self.config_loader.get_config()
        degradation_config = config.degradations
        
        if degradation_config and degradation_config.vst_plugin_path:
            return AudioDegradation(degradation_config.vst_plugin_path)
        return None

    def _initialize_audio_enhancers(self) -> Dict[str, AudioEnhancer]:
        """Initializes the audio enhancement modules if configured."""
        config = self.config_loader.get_config()
        enhancement_configs = config.enhancements
        
        enhancers = {}
        for enhancement_config in enhancement_configs:
            if enhancement_config.enabled:
                name = enhancement_config.name or enhancement_config.method
                if enhancement_config.method == "demucs":
                    enhancers[name] = DemucsEnhancer(enhancement_config.model_dump())
                elif enhancement_config.method == "deepfilternet":
                    enhancers[name] = DeepFilterNetEnhancer(enhancement_config.model_dump())
                else:
                    logger.warning(f"Unknown enhancement method: {enhancement_config.method}")
        return enhancers
