"""
Configuration Management System
Centralized configuration loader with environment-specific settings
Supports YAML configs, environment variables, and runtime overrides
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConfigValidationResult:
    """Configuration validation result"""
    is_valid: bool
    errors: list
    warnings: list
    config_summary: dict

class ConfigManager:
    """
    Production configuration manager with validation and environment handling
    Loads configurations from YAML files with environment variable overrides
    """
    
    def __init__(self, config_dir: str = "config", environment: str = None):
        self.config_dir = Path(config_dir)
        self.environment = environment or os.getenv('ENVIRONMENT', 'development')
        self.config_data = {}
        self.loaded_files = []
        
        # Default configuration files to load
        self.config_files = [
            "base.yaml",
            f"{self.environment}.yaml"
        ]
        
        # Load configurations
        self._load_configurations()
    
    def _load_configurations(self):
        """Load configuration files in priority order"""
        logger.info(f"🔧 Loading configurations for environment: {self.environment}")
        
        for config_file in self.config_files:
            config_path = self.config_dir / config_file
            
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        file_config = yaml.safe_load(f) or {}
                    
                    # Deep merge configurations
                    self._deep_merge(self.config_data, file_config)
                    self.loaded_files.append(str(config_path))
                    logger.info(f"✅ Loaded config: {config_file}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load {config_file}: {e}")
                    raise
            else:
                logger.warning(f"⚠️ Config file not found: {config_file}")
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        logger.info(f"📋 Loaded {len(self.loaded_files)} config files")
    
    def _deep_merge(self, target: dict, source: dict):
        """Deep merge source dictionary into target dictionary"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides using dot notation"""
        logger.info("🔄 Applying environment variable overrides...")
        
        # Common environment variable patterns
        env_mappings = {
            # Model serving
            'MODEL_HOST': 'model_serving.host',
            'MODEL_PORT': 'model_serving.port',
            'MODEL_WORKERS': 'model_serving.max_workers',
            
            # Database
            'DATABASE_URL': 'database.connection_string',
            'DATABASE_ENABLED': 'database.enable_database',
            
            # Redis
            'REDIS_URL': 'caching.redis_url',
            'REDIS_ENABLED': 'caching.enable_redis',
            
            # Monitoring
            'ALERT_EMAIL': 'monitoring.alert_email',
            'SLACK_WEBHOOK': 'monitoring.slack_webhook_url',
            
            # Security
            'JWT_SECRET': 'security.jwt_secret',
            'SSL_CERT_PATH': 'security.ssl_cert_path',
            'SSL_KEY_PATH': 'security.ssl_key_path',
            
            # Logging
            'LOG_LEVEL': 'logging.level',
            'LOG_FILE': 'logging.log_file',
            
            # Feature flags
            'ENABLE_AB_TESTING': 'feature_flags.enable_ab_testing',
            'ENABLE_EXPERIMENTAL_MODELS': 'feature_flags.enable_experimental_models',
        }
        
        override_count = 0
        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Type conversion
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif self._is_float(value):
                    value = float(value)
                
                self._set_nested_value(self.config_data, config_path, value)
                override_count += 1
                logger.info(f"📝 Override: {config_path} = {value}")
        
        if override_count > 0:
            logger.info(f"✅ Applied {override_count} environment overrides")
    
    def _is_float(self, value: str) -> bool:
        """Check if string can be converted to float"""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def _set_nested_value(self, data: dict, path: str, value: Any):
        """Set value at nested path using dot notation"""
        keys = path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation path
        
        Args:
            path: Dot-separated path to config value (e.g., 'models.default_model')
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        keys = path.split('.')
        current = self.config_data
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, path: str, value: Any):
        """Set configuration value using dot notation path"""
        self._set_nested_value(self.config_data, path, value)
    
    def get_section(self, section: str) -> dict:
        """Get entire configuration section"""
        return self.get(section, {})
    
    def validate(self) -> ConfigValidationResult:
        """Validate loaded configuration"""
        logger.info("🔍 Validating configuration...")
        
        errors = []
        warnings = []
        
        # Required configuration checks
        required_configs = [
            'environment.name',
            'models.default_model',
            'business_thresholds.min_precision_at_10',
            'business_thresholds.max_rmse'
        ]
        
        for required_path in required_configs:
            if self.get(required_path) is None:
                errors.append(f"Missing required configuration: {required_path}")
        
        # Business threshold validations
        thresholds = self.get_section('business_thresholds')
        if thresholds:
            if thresholds.get('min_precision_at_10', 0) > 1.0:
                errors.append("min_precision_at_10 cannot be greater than 1.0")
            
            if thresholds.get('max_rmse', 0) <= 0:
                errors.append("max_rmse must be positive")
            
            if thresholds.get('min_revenue_impact', 0) < 0:
                warnings.append("min_revenue_impact is negative - this may not be realistic")
        
        # Model configuration validations
        models_config = self.get_section('models')
        if models_config:
            supported_types = models_config.get('supported_types', [])
            default_model = models_config.get('default_model')
            
            if default_model and default_model not in supported_types:
                errors.append(f"default_model '{default_model}' not in supported_types")
        
        # Performance validations
        performance_config = self.get_section('performance')
        if performance_config:
            max_memory = performance_config.get('max_memory_usage_gb', 0)
            if max_memory > 32:
                warnings.append(f"max_memory_usage_gb ({max_memory}GB) is very high")
        
        # Security validations
        security_config = self.get_section('security')
        if security_config:
            if security_config.get('enable_https') and not security_config.get('ssl_cert_path'):
                errors.append("HTTPS enabled but no SSL certificate path configured")
        
        # Environment-specific validations
        if self.environment == 'production':
            # Production-specific requirements
            if not self.get('logging.enable_json_logging', False):
                warnings.append("JSON logging recommended for production")
            
            if not self.get('monitoring.enable_drift_detection', False):
                warnings.append("Drift detection recommended for production")
            
            if self.get('environment.debug', False):
                errors.append("Debug mode should not be enabled in production")
        
        # Generate configuration summary
        config_summary = {
            'environment': self.environment,
            'loaded_files': len(self.loaded_files),
            'total_configs': len(self._flatten_dict(self.config_data)),
            'default_model': self.get('models.default_model'),
            'monitoring_enabled': self.get('monitoring.enable_drift_detection', False),
            'security_features': {
                'https_enabled': self.get('security.enable_https', False),
                'auth_enabled': self.get('security.enable_auth', False),
                'rate_limiting': self.get('security.enable_rate_limiting', False)
            }
        }
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info(f"✅ Configuration validation passed ({len(warnings)} warnings)")
        else:
            logger.error(f"❌ Configuration validation failed ({len(errors)} errors)")
        
        return ConfigValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            config_summary=config_summary
        )
    
    def _flatten_dict(self, d: dict, parent_key: str = '', sep: str = '.') -> dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def export_config(self, output_path: str = None, format: str = 'yaml') -> str:
        """Export current configuration to file"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"config_export_{self.environment}_{timestamp}.{format}"
        
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(self.config_data, f, indent=2, default=str)
        else:  # yaml
            with open(output_path, 'w') as f:
                yaml.dump(self.config_data, f, default_flow_style=False)
        
        logger.info(f"📁 Configuration exported to: {output_path}")
        return output_path
    
    def reload(self):
        """Reload all configurations"""
        logger.info("🔄 Reloading configurations...")
        self.config_data = {}
        self.loaded_files = []
        self._load_configurations()
    
    def get_runtime_info(self) -> dict:
        """Get runtime configuration information"""
        return {
            'environment': self.environment,
            'config_dir': str(self.config_dir),
            'loaded_files': self.loaded_files,
            'load_time': datetime.now().isoformat(),
            'total_configs': len(self._flatten_dict(self.config_data)),
            'config_files_found': len(self.loaded_files),
            'env_overrides_applied': len([k for k in os.environ.keys() if k.startswith(('MODEL_', 'DATABASE_', 'REDIS_', 'LOG_'))]),
        }

# Global configuration instance
_config_instance = None

def get_config(config_dir: str = "config", environment: str = None, force_reload: bool = False) -> ConfigManager:
    """Get global configuration instance (singleton pattern)"""
    global _config_instance
    
    if _config_instance is None or force_reload:
        _config_instance = ConfigManager(config_dir=config_dir, environment=environment)
    
    return _config_instance

def main():
    """Test configuration management system"""
    print("🔧 Testing Configuration Management System...")
    
    # Initialize config manager
    config = ConfigManager()
    
    # Validate configuration
    validation_result = config.validate()
    
    print(f"\n📊 Configuration Summary:")
    print(f"Environment: {config.environment}")
    print(f"Valid: {validation_result.is_valid}")
    print(f"Errors: {len(validation_result.errors)}")
    print(f"Warnings: {len(validation_result.warnings)}")
    
    if validation_result.errors:
        print("\n❌ Errors:")
        for error in validation_result.errors:
            print(f"  - {error}")
    
    if validation_result.warnings:
        print("\n⚠️ Warnings:")
        for warning in validation_result.warnings:
            print(f"  - {warning}")
    
    # Test some configuration access
    print(f"\n🧪 Configuration Tests:")
    print(f"Default Model: {config.get('models.default_model')}")
    print(f"Max RMSE: {config.get('business_thresholds.max_rmse')}")
    print(f"Debug Mode: {config.get('environment.debug')}")
    print(f"Monitoring Enabled: {config.get('monitoring.enable_drift_detection')}")
    
    # Export configuration
    export_path = config.export_config()
    print(f"📁 Configuration exported to: {export_path}")
    
    print("\n✅ Configuration management system ready!")

if __name__ == "__main__":
    main()