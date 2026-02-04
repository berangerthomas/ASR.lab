import yaml
from pathlib import Path
from typing import Dict, Any

from ..core.models import BenchmarkConfig

class ConfigLoader:
    """
    Handles loading and validation of benchmark configuration files using Pydantic.
    """

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config: BenchmarkConfig = self._load_config()

    def _load_config(self) -> BenchmarkConfig:
        """Loads and validates the YAML configuration file."""
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            try:
                raw_config = yaml.safe_load(f)
                return BenchmarkConfig(**raw_config)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file: {e}")
            except Exception as e:
                raise ValueError(f"Configuration validation failed: {e}")

    def get_config(self) -> BenchmarkConfig:
        """Returns the validated configuration object."""
        return self.config

