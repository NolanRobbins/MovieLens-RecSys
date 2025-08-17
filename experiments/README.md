# Streamlined Experiment Management System

A unified, template-based experiment tracking system that simplifies configuration management, experiment execution, and results comparison.

## 🚀 Quick Start

### 1. Create an Experiment

```bash
# From a template
python experiments/experiment_manager.py create hybrid_vae_baseline my_experiment

# With modifications
python experiments/experiment_manager.py create hybrid_vae_baseline my_experiment \
  --modifications '{"training": {"learning_rate": 0.001, "batch_size": 2048}}'
```

### 2. Run the Experiment

```bash
python experiments/run_experiment.py my_experiment_20250817_143022
```

### 3. Monitor and Compare

```bash
# List all experiments
python experiments/experiment_manager.py list

# Compare experiments
python experiments/experiment_manager.py compare exp1_20250817_143022 exp2_20250817_144533

# Search experiments
python experiments/experiment_manager.py list --status completed --tag baseline
```

## 📁 Directory Structure

```
experiments/
├── experiment_manager.py      # Core management system
├── run_experiment.py         # Unified experiment runner
├── templates/                # Experiment templates
│   ├── hybrid_vae_baseline.yaml
│   ├── hybrid_vae_advanced.yaml
│   └── neural_cf_baseline.yaml
├── configs/                  # Generated experiment configs
│   ├── exp1_20250817_143022.yaml
│   └── exp2_20250817_144533.yaml
├── results/                  # Experiment results
│   ├── exp1_20250817_143022/
│   │   ├── results.json
│   │   ├── best_model.pt
│   │   └── logs/
│   └── exp2_20250817_144533/
├── registry.jsonl           # Experiment registry (append-only log)
└── status.json             # Current experiment status
```

## 🎯 Key Features

### 1. Template-Based Configuration
- **Reusable templates** for different model types
- **Easy customization** with YAML modifications
- **Version control** for experiment configurations

### 2. Unified Experiment Tracking
- **Automatic naming** with timestamps
- **Status tracking** (planned → running → completed/failed)
- **Append-only registry** for complete experiment history

### 3. Easy Comparison and Search
- **Pandas DataFrames** for experiment analysis
- **Flexible filtering** by status, tags, metrics
- **Side-by-side comparison** of configurations

### 4. Integration Ready
- **W&B integration** built-in
- **Model versioning** compatible with existing system
- **CLI and Python API** for automation

## 📋 Available Templates

### 1. `hybrid_vae_baseline.yaml`
Standard Hybrid VAE configuration
- **Target RMSE**: 0.75
- **Architecture**: [512, 256], 64 latent dim
- **Training**: Adam, 100 epochs, batch size 1024

### 2. `hybrid_vae_advanced.yaml`
Advanced Hybrid VAE with regularization
- **Target RMSE**: 0.55
- **Architecture**: [1024, 512, 256, 128], 128 latent dim
- **Training**: AdamW, 200 epochs, batch size 2048
- **Features**: Label smoothing, mixup, cosine scheduling

### 3. `neural_cf_baseline.yaml`
Neural Collaborative Filtering baseline
- **Target RMSE**: 0.80
- **Architecture**: [128, 64, 32], 64 embedding dim
- **Training**: Adam, 50 epochs, batch size 1024

## 🔧 Usage Examples

### Python API

```python
from experiments.experiment_manager import ExperimentManager

manager = ExperimentManager()

# Create experiment from template
exp_id = manager.create_experiment_from_template(
    template_name="hybrid_vae_advanced",
    experiment_name="high_regularization_test",
    modifications={
        "training": {
            "dropout_rate": 0.5,
            "weight_decay": 0.001
        }
    }
)

# Start experiment
manager.start_experiment(exp_id)

# Complete with results
manager.complete_experiment(
    exp_id,
    results={"rmse": 0.52, "precision_at_10": 0.41},
    model_path="path/to/model.pt"
)

# Get best experiment
best_exp = manager.get_best_experiment(metric="rmse")
print(f"Best experiment: {best_exp}")

# Clone successful experiment
new_exp = manager.clone_experiment(
    source_experiment_id=best_exp,
    new_name="refined_version",
    modifications={"training": {"learning_rate": 0.0001}}
)
```

### Command Line Interface

```bash
# List experiments with filters
python experiments/experiment_manager.py list --status completed

# Create experiment with specific modifications
python experiments/experiment_manager.py create hybrid_vae_advanced my_test \
  --modifications '{"training": {"learning_rate": 0.001, "batch_size": 4096}}'

# Compare multiple experiments
python experiments/experiment_manager.py compare \
  exp1_20250817_143022 \
  exp2_20250817_144533 \
  exp3_20250817_145044

# Run experiment
python experiments/run_experiment.py my_test_20250817_143022

# Show experiment config (dry run)
python experiments/run_experiment.py my_test_20250817_143022 --dry-run
```

## 📊 Experiment Tracking

### Registry Format (JSONL)
```json
{"experiment_id": "exp1_20250817_143022", "name": "baseline_test", "template": "hybrid_vae_baseline", "status": "planned", "created_at": "2025-08-17T14:30:22"}
{"experiment_id": "exp1_20250817_143022", "event": "started", "timestamp": "2025-08-17T14:31:00"}
{"experiment_id": "exp1_20250817_143022", "event": "completed", "timestamp": "2025-08-17T16:45:30", "results": {"rmse": 0.68}}
```

### Status Tracking
```json
{
  "exp1_20250817_143022": {
    "status": "completed",
    "created_at": "2025-08-17T14:30:22",
    "started_at": "2025-08-17T14:31:00",
    "completed_at": "2025-08-17T16:45:30",
    "results": {"rmse": 0.68, "precision_at_10": 0.35}
  }
}
```

## 🔄 Migration from Current System

### 1. Import Existing Experiments
```python
# Import from current experiment_registry.json
from experiments.experiment_manager import ExperimentManager
import json

manager = ExperimentManager()

# Load existing registry
with open('models/experiment_registry.json', 'r') as f:
    old_registry = json.load(f)

# Convert to new format
for exp_id, exp_data in old_registry.items():
    # Create experiment config from old data
    # Register in new system
```

### 2. Use Existing W&B Integration
The new system works with your existing W&B setup - just update the project names in templates.

### 3. Gradual Adoption
- **Start with new experiments** using templates
- **Keep existing scripts** for running models
- **Gradually migrate** successful patterns to templates

## 🎯 Benefits Over Current System

1. **Reduced Manual Work**: No more manually editing experiment files
2. **Better Organization**: Clear separation of templates, configs, and results
3. **Easy Comparison**: Built-in experiment comparison tools
4. **Reproducibility**: Version-controlled configurations
5. **Automation Ready**: CLI and API for CI/CD integration
6. **Search & Filter**: Find experiments by any criteria
7. **Clone & Modify**: Build on successful experiments easily

## 🚀 Next Steps

1. **Create your first experiment** from a template
2. **Run a quick test** to validate the system
3. **Migrate successful patterns** from your current experiments
4. **Customize templates** for your specific needs
5. **Integrate with CI/CD** for automated experimentation