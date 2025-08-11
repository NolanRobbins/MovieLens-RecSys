# Phase 3: CI/CD Production Pipeline - Deployment Guide

## Overview

Phase 3 implements a comprehensive CI/CD production pipeline for the MovieLens recommendation system with automated training, validation, deployment, and rollback capabilities.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   GitHub Actions │────│  Model Training  │────│   Validation    │
│   CI/CD Pipeline │    │   & Evaluation   │    │   & Quality     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                        │
         ▼                       ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Drift Detection │    │ Model Versioning │    │  Streamlit      │
│ & Monitoring    │    │ & Rollback Sys   │    │  Cloud Deploy   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Components Implemented

### 1. GitHub Actions Workflows

#### CI Workflow (`.github/workflows/ci.yml`)
- **Purpose**: Code quality, security scanning, and basic validation
- **Triggers**: Push to main/develop, pull requests
- **Features**:
  - Python linting and formatting
  - Security vulnerability scanning
  - Documentation validation
  - Basic import and structure tests

#### Model Training Pipeline (`.github/workflows/model-training.yml`)
- **Purpose**: Automated model training and validation
- **Triggers**: Manual dispatch, scheduled runs, drift detection events
- **Features**:
  - Multi-model training (Matrix Factorization, Neural CF, Two-Tower, Hybrid VAE)
  - Business threshold validation
  - Automated deployment decisions
  - Artifact management
  - Performance comparison

#### Drift Detection Workflow (`.github/workflows/drift-detection.yml`)
- **Purpose**: Daily model performance monitoring and drift detection
- **Triggers**: Daily schedule (6 AM UTC), manual dispatch
- **Features**:
  - Statistical drift analysis
  - Data quality monitoring
  - Automated retraining triggers
  - GitHub issue creation for alerts
  - Configurable sensitivity levels

#### Model Rollback System (`.github/workflows/model-rollback.yml`)
- **Purpose**: Safe model rollback with validation
- **Triggers**: Manual dispatch with confirmation
- **Features**:
  - Pre-rollback health checks
  - Automated rollback execution
  - Post-rollback validation
  - Stakeholder notifications

### 2. Model Validation System (`src/validation/model_validator.py`)

#### Business Threshold Validation
- **RMSE**: < 0.85 (critical)
- **Precision@10**: > 0.35 (critical)
- **NDCG@10**: > 0.42 (critical)
- **Diversity**: > 50% (warning)
- **Coverage**: > 60% (warning)
- **Revenue Impact**: > $500K annually (critical)
- **Inference Time**: < 50ms p95 (critical)

#### Validation Features
- Automated model loading and testing
- Business impact assessment
- Operational requirement checks
- Data quality validation
- Training stability analysis
- Comprehensive reporting

### 3. Model Versioning & Rollback (`src/deployment/model_versioning.py`)

#### Version Management
- Unique version IDs with timestamps
- Model integrity verification (SHA256 hashing)
- Deployment status tracking
- Metadata preservation (training config, validation results, business metrics)

#### Rollback Capabilities
- Automatic rollback to last stable version
- Manual rollback to specific versions
- Health checks before rollback
- Rollback reason tracking
- Post-rollback validation

#### Features
- Model lifecycle management
- Disk usage optimization
- Health monitoring
- Rollback history tracking

### 4. Production Configuration System

#### Configuration Manager (`src/config/config_manager.py`)
- Environment-specific configurations
- Environment variable overrides
- Configuration validation
- Runtime configuration reloading

#### Configuration Files
- `config/base.yaml`: Common settings
- `config/development.yaml`: Development overrides
- `config/production.yaml`: Production settings

### 5. Streamlit Cloud Deployment

#### Deployment Configuration
- `.streamlit/config.toml`: Production Streamlit settings
- `.streamlit/secrets.toml`: Secrets management template
- Environment-specific theming and performance settings

## 📊 Business Metrics Integration

### Key Performance Indicators
- **Precision@10**: Recommendation accuracy
- **NDCG@10**: Ranking quality
- **RMSE**: Rating prediction accuracy
- **Diversity**: Recommendation variety
- **Coverage**: Item catalog coverage

### Business Impact Metrics
- **Revenue Impact**: Projected annual revenue change
- **CTR Improvement**: Click-through rate enhancement
- **Session Duration**: User engagement time
- **Inference Time**: Model response latency

## 🔄 CI/CD Pipeline Flow

### 1. Code Changes
```
Developer Push → CI Workflow → Code Quality Checks → Tests Pass
```

### 2. Model Training
```
Trigger Training → Data Validation → Multi-Model Training → 
Business Validation → Deployment Decision → Model Versioning
```

### 3. Drift Detection
```
Daily Schedule → Data Quality Check → Statistical Analysis → 
Drift Detection → Alert Generation → Auto-Retraining (if needed)
```

### 4. Model Rollback
```
Manual Trigger → Pre-Rollback Checks → Execute Rollback → 
Post-Validation → Stakeholder Notification
```

## 🚦 Quality Gates

### Critical Thresholds (Must Pass)
- RMSE < 0.85
- Precision@10 > 0.35
- NDCG@10 > 0.42
- Revenue Impact > $500K
- Inference Time < 50ms
- Model Loading Success

### Warning Thresholds (Review Required)
- Diversity < 50%
- Coverage < 60%
- Model Size > 1GB
- Training Time > 2 hours

## 🔧 Configuration Management

### Environment Variables
```bash
# Model Serving
MODEL_HOST=0.0.0.0
MODEL_PORT=8501
MODEL_WORKERS=4

# Database (if enabled)
DATABASE_URL=postgresql://user:pass@host:port/db
DATABASE_ENABLED=true

# Monitoring
ALERT_EMAIL=ml-team@company.com
SLACK_WEBHOOK=https://hooks.slack.com/...

# Security
JWT_SECRET=your-secret-key
SSL_CERT_PATH=/etc/ssl/cert.pem
```

### Configuration Sections
- `environment`: Environment-specific settings
- `models`: Model configuration and paths
- `business_thresholds`: Quality gates and thresholds
- `monitoring`: Drift detection and alerting
- `training`: Model training parameters
- `security`: Authentication and encryption
- `performance`: Resource limits and optimization

## 📈 Monitoring & Alerting

### Drift Detection
- **Statistical Tests**: KS test, chi-square, drift scores
- **Data Quality**: Missing values, distribution changes
- **Performance Degradation**: Metric threshold breaches
- **Automated Actions**: Retraining triggers, alerts

### Alert Channels
- **GitHub Issues**: Automated issue creation
- **Email Notifications**: Stakeholder alerts
- **Slack Integration**: Real-time notifications
- **Dashboard Updates**: Streamlit app status

## 🔒 Security & Compliance

### Security Features
- HTTPS/TLS encryption
- Rate limiting and CORS
- JWT authentication (configurable)
- Secrets management
- Input validation

### Compliance Features
- Audit logging
- Data lineage tracking
- Model explainability (configurable)
- Data retention policies
- Backup and disaster recovery

## 🚀 Deployment Instructions

### 1. Repository Setup
```bash
# Clone the repository
git clone https://github.com/NolanRobbins/MovieLens-RecSys.git
cd MovieLens-RecSys

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Copy and customize configuration
cp config/production.yaml.example config/production.yaml

# Set environment variables
export ENVIRONMENT=production
export MODEL_HOST=0.0.0.0
export ALERT_EMAIL=your-email@company.com
```

### 3. GitHub Actions Setup
1. Enable GitHub Actions in repository settings
2. Add required secrets in repository settings:
   - `GITHUB_TOKEN` (automatically provided)
   - Add any additional secrets for external services

### 4. Streamlit Cloud Deployment
1. Connect repository to Streamlit Cloud
2. Configure secrets in Streamlit dashboard
3. Set main file path: `streamlit_app.py`
4. Deploy application

### 5. Model Training
```bash
# Manual training trigger
gh workflow run model-training.yml

# Or use the GitHub Actions UI
```

## 📋 Maintenance & Operations

### Regular Tasks
- **Daily**: Automatic drift detection runs
- **Weekly**: Review validation reports and performance metrics
- **Monthly**: Model version cleanup and optimization
- **Quarterly**: Configuration review and threshold tuning

### Troubleshooting
- Check GitHub Actions logs for workflow failures
- Review drift detection reports for performance issues
- Validate configuration using built-in validation
- Monitor resource usage and adjust limits as needed

### Performance Optimization
- Adjust model cache size based on memory availability
- Tune batch sizes for optimal throughput
- Configure worker counts based on CPU cores
- Monitor and adjust timeout values

## 🎯 Success Metrics

### Operational Metrics
- **Deployment Success Rate**: > 95%
- **Rollback Time**: < 5 minutes
- **Model Validation Time**: < 10 minutes
- **Pipeline Execution Time**: < 30 minutes

### Business Metrics
- **Model Performance**: Meet all critical thresholds
- **System Uptime**: > 99.9%
- **Alert Response Time**: < 1 hour
- **Issue Resolution Time**: < 24 hours

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Streamlit Cloud Deployment](https://docs.streamlit.io/streamlit-cloud)
- [MLOps Best Practices](https://ml-ops.org/)
- [Model Monitoring Guide](docs/monitoring_guide.md)

## 🤝 Support

For questions or issues with the Phase 3 deployment:
1. Check the troubleshooting section above
2. Review GitHub Actions logs
3. Create an issue in the repository
4. Contact the ML engineering team

---

**Phase 3 Status**: ✅ Complete - Production CI/CD pipeline fully implemented and ready for deployment.