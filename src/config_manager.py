import yaml
import os
from typing import Dict, Any
import logging

class ConfigManager:
    """Configuration manager for loading and managing YAML config files."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._ensure_numeric_types()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            logging.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def _ensure_numeric_types(self):
        """Ensure numeric values are properly typed."""
        numeric_keys = [
            'dataset.num_classes', 'dataset.batch_size', 'dataset.num_workers',
            'model.input_channels', 'model.num_conv_blocks', 'model.kernel_size',
            'model.padding', 'model.pool_size', 'model.pool_stride', 'model.dropout_rate',
            'model.fc_hidden_size', 'training.epochs', 'training.learning_rate',
            'training.weight_decay', 'training.scheduler.step_size', 'training.scheduler.gamma',
            'visualization.num_sample_predictions', 'random_seed'
        ]
        
        for key in numeric_keys:
            value = self.get(key)
            if value is not None:
                # Convert to appropriate numeric type
                if isinstance(value, str):
                    try:
                        # Try float first (handles scientific notation)
                        converted = float(value)
                        # If it's a whole number, convert to int
                        if converted.is_integer():
                            converted = int(converted)
                        self.set(key, converted)
                        logging.debug(f"Converted {key}: {value} -> {converted}")
                    except ValueError:
                        logging.warning(f"Could not convert {key}: {value} to numeric")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'model.num_conv_blocks')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'model.num_conv_blocks')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: str = None) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save the configuration (default: original config path)
        """
        save_path = path or self.config_path
        
        try:
            with open(save_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config, file, default_flow_style=False, indent=2)
            logging.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
            raise
    
    def create_directories(self) -> None:
        """Create necessary directories based on configuration."""
        directories = [
            self.get('evaluation.metrics_file', '').rsplit('/', 1)[0],
            self.get('visualization.plots_dir'),
            self.get('hyperparameter_analysis.results_dir'),
            self.get('logging.model_save_dir'),
            self.get('logging.log_file', '').rsplit('/', 1)[0]
        ]
        
        for directory in directories:
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logging.info(f"Created directory: {directory}")
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return yaml.dump(self.config, default_flow_style=False, indent=2)
