import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    """
    Handles loading and validation of benchmark configuration files.
    """

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Loads the YAML configuration file."""
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file: {e}")

    def get_section(self, section_name: str) -> Dict[str, Any]:
        """Returns a specific section of the configuration."""
        return self.config.get(section_name, {})

    def get_parameter(self, section_name: str, param_name: str, default: Any = None) -> Any:
        """Returns a specific parameter from a section."""
        return self.get_section(section_name).get(param_name, default)
