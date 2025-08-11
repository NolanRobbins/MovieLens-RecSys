"""
Model Versioning and Rollback System
Production-grade model version control with rollback capabilities
Manages model lifecycle, deployment history, and rollback procedures
"""

import torch
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentStatus(Enum):
    """Model deployment status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ROLLBACK = "rollback"
    STAGING = "staging"
    FAILED = "failed"
    RETIRED = "retired"

class RollbackReason(Enum):
    """Reasons for model rollback"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    BUSINESS_IMPACT = "business_impact"
    TECHNICAL_FAILURE = "technical_failure"
    MANUAL_INTERVENTION = "manual_intervention"
    DRIFT_DETECTION = "drift_detection"
    VALIDATION_FAILURE = "validation_failure"

@dataclass
class ModelVersion:
    """Model version metadata"""
    version_id: str
    model_name: str
    model_path: str
    deployment_timestamp: datetime
    status: DeploymentStatus
    git_sha: Optional[str] = None
    training_config: Optional[Dict[str, Any]] = None
    validation_results: Optional[Dict[str, Any]] = None
    business_metrics: Optional[Dict[str, Any]] = None
    model_hash: Optional[str] = None
    rollback_version: Optional[str] = None
    deployment_notes: Optional[str] = None

@dataclass
class RollbackEvent:
    """Model rollback event record"""
    rollback_id: str
    from_version: str
    to_version: str
    reason: RollbackReason
    triggered_by: str
    timestamp: datetime
    success: bool
    rollback_notes: str
    impact_assessment: Optional[Dict[str, Any]] = None

class ModelVersionManager:
    """
    Production model versioning system with rollback capabilities
    Manages model deployment lifecycle and provides rollback functionality
    """
    
    def __init__(self, models_dir: str = "data/models", versions_dir: str = "data/model_versions"):
        self.models_dir = Path(models_dir)
        self.versions_dir = Path(versions_dir)
        self.metadata_file = self.versions_dir / "version_metadata.json"
        self.rollback_log_file = self.versions_dir / "rollback_history.json"
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata
        self.version_history: List[ModelVersion] = self._load_version_metadata()
        self.rollback_history: List[RollbackEvent] = self._load_rollback_history()
    
    def deploy_model(self, model_path: str, model_name: str, 
                    validation_results: Optional[Dict[str, Any]] = None,
                    training_config: Optional[Dict[str, Any]] = None,
                    business_metrics: Optional[Dict[str, Any]] = None,
                    git_sha: Optional[str] = None,
                    deployment_notes: Optional[str] = None) -> str:
        """
        Deploy a new model version
        
        Args:
            model_path: Path to model checkpoint
            model_name: Name of the model
            validation_results: Model validation results
            training_config: Training configuration used
            business_metrics: Business impact metrics
            git_sha: Git commit SHA
            deployment_notes: Optional deployment notes
            
        Returns:
            Version ID of deployed model
        """
        logger.info(f"🚀 Deploying new model version: {model_name}")
        
        # Generate version ID
        version_id = self._generate_version_id(model_name)
        
        # Calculate model hash for integrity
        model_hash = self._calculate_model_hash(model_path)
        
        # Create versioned model directory
        version_dir = self.versions_dir / version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model to versioned storage
        versioned_model_path = version_dir / f"{model_name}.pt"
        shutil.copy2(model_path, versioned_model_path)
        
        # Deactivate current active versions
        self._deactivate_current_versions(model_name)
        
        # Create model version metadata
        model_version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            model_path=str(versioned_model_path),
            deployment_timestamp=datetime.now(),
            status=DeploymentStatus.ACTIVE,
            git_sha=git_sha,
            training_config=training_config,
            validation_results=validation_results,
            business_metrics=business_metrics,
            model_hash=model_hash,
            deployment_notes=deployment_notes
        )
        
        # Save version metadata
        self.version_history.append(model_version)
        self._save_version_metadata()
        
        # Save deployment configuration
        self._save_deployment_config(version_id, model_version)
        
        # Update active model symlink
        self._update_active_model_link(model_name, versioned_model_path)
        
        logger.info(f"✅ Model deployed successfully: {version_id}")
        return version_id
    
    def rollback_model(self, model_name: str, target_version: Optional[str] = None,
                      reason: RollbackReason = RollbackReason.MANUAL_INTERVENTION,
                      triggered_by: str = "system",
                      rollback_notes: str = "") -> bool:
        """
        Rollback model to previous stable version
        
        Args:
            model_name: Name of model to rollback
            target_version: Specific version to rollback to (if None, uses last stable)
            reason: Reason for rollback
            triggered_by: Who/what triggered the rollback
            rollback_notes: Additional notes about rollback
            
        Returns:
            True if rollback successful
        """
        logger.info(f"🔄 Initiating rollback for model: {model_name}")
        
        # Get current active version
        current_version = self.get_active_version(model_name)
        if not current_version:
            logger.error(f"❌ No active version found for model: {model_name}")
            return False
        
        # Determine target version
        if target_version:
            target_model = next((v for v in self.version_history 
                               if v.version_id == target_version and v.model_name == model_name), None)
        else:
            # Get last stable version (excluding current)
            stable_versions = [
                v for v in self.version_history 
                if (v.model_name == model_name and 
                    v.version_id != current_version.version_id and
                    v.status not in [DeploymentStatus.FAILED, DeploymentStatus.RETIRED])
            ]
            target_model = max(stable_versions, key=lambda x: x.deployment_timestamp) if stable_versions else None
        
        if not target_model:
            logger.error(f"❌ No suitable rollback target found for model: {model_name}")
            return False
        
        rollback_id = f"rollback_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Validate target model exists and is accessible
            if not Path(target_model.model_path).exists():
                logger.error(f"❌ Target model file not found: {target_model.model_path}")
                return False
            
            # Verify model integrity
            if not self._verify_model_integrity(target_model):
                logger.error(f"❌ Model integrity check failed for: {target_model.version_id}")
                return False
            
            # Update model statuses
            current_version.status = DeploymentStatus.ROLLBACK
            target_model.status = DeploymentStatus.ACTIVE
            
            # Update active model symlink
            self._update_active_model_link(model_name, target_model.model_path)
            
            # Record rollback event
            rollback_event = RollbackEvent(
                rollback_id=rollback_id,
                from_version=current_version.version_id,
                to_version=target_model.version_id,
                reason=reason,
                triggered_by=triggered_by,
                timestamp=datetime.now(),
                success=True,
                rollback_notes=rollback_notes
            )
            
            self.rollback_history.append(rollback_event)
            
            # Save updated metadata
            self._save_version_metadata()
            self._save_rollback_history()
            
            logger.info(f"✅ Rollback successful: {current_version.version_id} → {target_model.version_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            
            # Record failed rollback
            rollback_event = RollbackEvent(
                rollback_id=rollback_id,
                from_version=current_version.version_id,
                to_version=target_model.version_id if target_model else "unknown",
                reason=reason,
                triggered_by=triggered_by,
                timestamp=datetime.now(),
                success=False,
                rollback_notes=f"Rollback failed: {str(e)}"
            )
            
            self.rollback_history.append(rollback_event)
            self._save_rollback_history()
            
            return False
    
    def get_active_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get currently active version for a model"""
        active_versions = [
            v for v in self.version_history 
            if v.model_name == model_name and v.status == DeploymentStatus.ACTIVE
        ]
        return active_versions[0] if active_versions else None
    
    def list_versions(self, model_name: Optional[str] = None, 
                     limit: Optional[int] = None) -> List[ModelVersion]:
        """List model versions with optional filtering"""
        versions = self.version_history
        
        if model_name:
            versions = [v for v in versions if v.model_name == model_name]
        
        # Sort by deployment timestamp (newest first)
        versions.sort(key=lambda x: x.deployment_timestamp, reverse=True)
        
        if limit:
            versions = versions[:limit]
        
        return versions
    
    def get_version_info(self, version_id: str) -> Optional[ModelVersion]:
        """Get detailed information about a specific version"""
        return next((v for v in self.version_history if v.version_id == version_id), None)
    
    def retire_old_versions(self, model_name: str, keep_count: int = 5) -> int:
        """Retire old model versions, keeping only the most recent ones"""
        model_versions = [v for v in self.version_history if v.model_name == model_name]
        model_versions.sort(key=lambda x: x.deployment_timestamp, reverse=True)
        
        retired_count = 0
        for version in model_versions[keep_count:]:
            if version.status != DeploymentStatus.ACTIVE:
                version.status = DeploymentStatus.RETIRED
                retired_count += 1
                
                # Optionally remove old model files to save space
                if Path(version.model_path).exists():
                    Path(version.model_path).unlink()
                    logger.info(f"🗑️ Retired version: {version.version_id}")
        
        if retired_count > 0:
            self._save_version_metadata()
        
        return retired_count
    
    def get_rollback_history(self, model_name: Optional[str] = None, 
                           limit: Optional[int] = None) -> List[RollbackEvent]:
        """Get rollback history with optional filtering"""
        history = self.rollback_history
        
        if model_name:
            # Filter by model name (need to match with version history)
            model_version_ids = {v.version_id for v in self.version_history if v.model_name == model_name}
            history = [h for h in history if h.from_version in model_version_ids or h.to_version in model_version_ids]
        
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x.timestamp, reverse=True)
        
        if limit:
            history = history[:limit]
        
        return history
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check on versioning system"""
        logger.info("🔍 Running model versioning health check...")
        
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'total_versions': len(self.version_history),
            'active_models': {},
            'integrity_issues': [],
            'disk_usage_mb': 0,
            'rollback_stats': {},
            'recommendations': []
        }
        
        # Check active models
        active_models = {}
        for version in self.version_history:
            if version.status == DeploymentStatus.ACTIVE:
                active_models[version.model_name] = version.version_id
        
        health_report['active_models'] = active_models
        
        # Check model integrity
        integrity_issues = []
        total_size = 0
        
        for version in self.version_history:
            if version.status != DeploymentStatus.RETIRED:
                model_path = Path(version.model_path)
                if model_path.exists():
                    total_size += model_path.stat().st_size
                    
                    # Verify hash if available
                    if version.model_hash and not self._verify_model_integrity(version):
                        integrity_issues.append({
                            'version_id': version.version_id,
                            'issue': 'hash_mismatch',
                            'path': str(model_path)
                        })
                else:
                    integrity_issues.append({
                        'version_id': version.version_id,
                        'issue': 'file_missing',
                        'path': str(model_path)
                    })
        
        health_report['integrity_issues'] = integrity_issues
        health_report['disk_usage_mb'] = total_size / (1024 * 1024)
        
        # Rollback statistics
        successful_rollbacks = sum(1 for r in self.rollback_history if r.success)
        failed_rollbacks = sum(1 for r in self.rollback_history if not r.success)
        
        health_report['rollback_stats'] = {
            'total_rollbacks': len(self.rollback_history),
            'successful': successful_rollbacks,
            'failed': failed_rollbacks,
            'success_rate': successful_rollbacks / len(self.rollback_history) if self.rollback_history else 1.0
        }
        
        # Generate recommendations
        recommendations = []
        
        if len(integrity_issues) > 0:
            recommendations.append("Fix model integrity issues - some model files are missing or corrupted")
        
        if health_report['disk_usage_mb'] > 10000:  # 10GB
            recommendations.append("Consider retiring old model versions to reduce disk usage")
        
        model_counts = {}
        for version in self.version_history:
            model_counts[version.model_name] = model_counts.get(version.model_name, 0) + 1
        
        for model_name, count in model_counts.items():
            if count > 10:
                recommendations.append(f"Model '{model_name}' has {count} versions - consider cleanup")
        
        health_report['recommendations'] = recommendations
        
        logger.info(f"✅ Health check completed - {len(integrity_issues)} issues found")
        return health_report
    
    def _generate_version_id(self, model_name: str) -> str:
        """Generate unique version ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_name}_v{timestamp}"
    
    def _calculate_model_hash(self, model_path: str) -> str:
        """Calculate SHA256 hash of model file"""
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _verify_model_integrity(self, model_version: ModelVersion) -> bool:
        """Verify model file integrity using hash"""
        if not model_version.model_hash:
            return True  # No hash to verify against
        
        try:
            current_hash = self._calculate_model_hash(model_version.model_path)
            return current_hash == model_version.model_hash
        except Exception as e:
            logger.error(f"Hash verification failed: {e}")
            return False
    
    def _deactivate_current_versions(self, model_name: str):
        """Deactivate currently active versions of a model"""
        for version in self.version_history:
            if version.model_name == model_name and version.status == DeploymentStatus.ACTIVE:
                version.status = DeploymentStatus.INACTIVE
    
    def _update_active_model_link(self, model_name: str, model_path: str):
        """Update symlink to active model"""
        link_path = self.models_dir / f"{model_name}_active.pt"
        
        # Remove existing symlink
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        
        # Create new symlink
        link_path.symlink_to(Path(model_path).resolve())
    
    def _save_deployment_config(self, version_id: str, model_version: ModelVersion):
        """Save deployment configuration for a version"""
        version_dir = self.versions_dir / version_id
        config_file = version_dir / "deployment_config.yaml"
        
        config = {
            'version_id': version_id,
            'model_name': model_version.model_name,
            'deployment_timestamp': model_version.deployment_timestamp.isoformat(),
            'git_sha': model_version.git_sha,
            'training_config': model_version.training_config,
            'validation_results': model_version.validation_results,
            'business_metrics': model_version.business_metrics,
            'deployment_notes': model_version.deployment_notes
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def _load_version_metadata(self) -> List[ModelVersion]:
        """Load version metadata from file"""
        if not self.metadata_file.exists():
            return []
        
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            versions = []
            for item in data:
                # Convert timestamp string back to datetime
                item['deployment_timestamp'] = datetime.fromisoformat(item['deployment_timestamp'])
                item['status'] = DeploymentStatus(item['status'])
                versions.append(ModelVersion(**item))
            
            return versions
            
        except Exception as e:
            logger.error(f"Error loading version metadata: {e}")
            return []
    
    def _save_version_metadata(self):
        """Save version metadata to file"""
        try:
            data = []
            for version in self.version_history:
                version_dict = asdict(version)
                version_dict['deployment_timestamp'] = version.deployment_timestamp.isoformat()
                version_dict['status'] = version.status.value
                data.append(version_dict)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving version metadata: {e}")
    
    def _load_rollback_history(self) -> List[RollbackEvent]:
        """Load rollback history from file"""
        if not self.rollback_log_file.exists():
            return []
        
        try:
            with open(self.rollback_log_file, 'r') as f:
                data = json.load(f)
            
            events = []
            for item in data:
                item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                item['reason'] = RollbackReason(item['reason'])
                events.append(RollbackEvent(**item))
            
            return events
            
        except Exception as e:
            logger.error(f"Error loading rollback history: {e}")
            return []
    
    def _save_rollback_history(self):
        """Save rollback history to file"""
        try:
            data = []
            for event in self.rollback_history:
                event_dict = asdict(event)
                event_dict['timestamp'] = event.timestamp.isoformat()
                event_dict['reason'] = event.reason.value
                data.append(event_dict)
            
            with open(self.rollback_log_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving rollback history: {e}")

def main():
    """Test model versioning system"""
    print("🔧 Testing Model Versioning System...")
    
    # Initialize version manager
    version_manager = ModelVersionManager()
    
    # Test with existing models
    model_files = list(Path("data/models").glob("*.pt"))
    
    if model_files:
        test_model = model_files[0]
        model_name = test_model.stem
        
        print(f"Testing with model: {test_model}")
        
        # Deploy new version
        version_id = version_manager.deploy_model(
            model_path=str(test_model),
            model_name=model_name,
            deployment_notes="Test deployment from versioning system",
            validation_results={'rmse': 0.82, 'precision_at_10': 0.36}
        )
        
        print(f"✅ Deployed version: {version_id}")
        
        # List versions
        versions = version_manager.list_versions(model_name, limit=5)
        print(f"📋 Found {len(versions)} versions for {model_name}")
        
        # Health check
        health = version_manager.health_check()
        print(f"🏥 Health check: {len(health['integrity_issues'])} issues found")
        
    else:
        print("⚠️ No model files found for testing")
    
    print("✅ Model versioning system ready!")

if __name__ == "__main__":
    main()