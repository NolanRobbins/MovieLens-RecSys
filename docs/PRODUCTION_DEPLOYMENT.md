# 🚀 Production Deployment Guide

Complete guide for deploying the MovieLens Recommendation System to production environments with monitoring, scaling, and security best practices.

## 📋 **Pre-Deployment Checklist**

### ✅ **System Validation**
```bash
# Validate all systems are working
python validate_system.py

# Run comprehensive tests
python -m pytest tests/ -v

# Check system health
python -c "
from src.config.config_manager import ConfigManager
config = ConfigManager()
validation = config.validate()
print(f'✅ Config Valid: {validation.is_valid}')
"
```

### 🔧 **Environment Setup**
```bash
# Production environment variables
export ENVIRONMENT=production
export MODEL_HOST=0.0.0.0
export MODEL_PORT=8501
export ALERT_EMAIL=ml-team@yourcompany.com
export JWT_SECRET=your-production-jwt-secret
export DATABASE_URL=postgresql://user:password@host:port/database
```

## 🌐 **Deployment Option 1: Streamlit Cloud**

### **Step 1: Repository Setup**
Your repository is already configured at:
```
https://github.com/NolanRobbins/MovieLens-RecSys
```

### **Step 2: Streamlit Cloud Deployment**
1. **Visit**: [share.streamlit.io](https://share.streamlit.io)
2. **Connect GitHub**: Link your GitHub account
3. **Deploy App**:
   - Repository: `NolanRobbins/MovieLens-RecSys`
   - Branch: `main`
   - Main file: `streamlit_app.py`
   - Python version: `3.9`

### **Step 3: Configure Secrets**
In Streamlit Cloud dashboard, add these secrets:
```toml
[secrets]
ENVIRONMENT = "production"
MODEL_HOST = "0.0.0.0"
ALERT_EMAIL = "ml-team@yourcompany.com"
JWT_SECRET = "your-jwt-secret"

[database]
host = "your-db-host"
port = 5432
database = "movielens"
username = "ml_user"
password = "secure-password"
```

### **Step 4: Health Monitoring**
```python
# Add this to your Streamlit app for production monitoring
import streamlit as st
from datetime import datetime

if st.sidebar.button("🏥 System Health Check"):
    try:
        # Run validation
        success = subprocess.run(['python', 'validate_system.py'], 
                               capture_output=True, text=True)
        if success.returncode == 0:
            st.success("✅ All systems healthy")
        else:
            st.error("❌ System issues detected")
            st.code(success.stderr)
    except Exception as e:
        st.error(f"Health check failed: {e}")
```

## 🐳 **Deployment Option 2: Docker Production**

### **Step 1: Production Dockerfile**
```dockerfile
# Production Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### **Step 2: Production Docker Compose**
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  movielens-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - ENVIRONMENT=production
      - MODEL_HOST=0.0.0.0
      - MODEL_PORT=8501
    volumes:
      - ./data:/app/data:ro
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - movielens-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped
    networks:
      - movielens-network

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
      - '--web.enable-lifecycle'
    networks:
      - movielens-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    networks:
      - movielens-network

volumes:
  redis-data:
  prometheus-data:
  grafana-data:

networks:
  movielens-network:
    driver: bridge
```

### **Step 3: Deploy with Docker**
```bash
# Build and deploy
docker-compose -f docker-compose.prod.yml up --build -d

# Monitor logs
docker-compose -f docker-compose.prod.yml logs -f

# Scale application
docker-compose -f docker-compose.prod.yml up --scale movielens-app=3 -d
```

## ☁️ **Deployment Option 3: AWS Cloud**

### **Step 1: AWS ECS Deployment**
```json
{
  "family": "movielens-recsys",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "movielens-app",
      "image": "your-account.dkr.ecr.region.amazonaws.com/movielens-recsys:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "ENVIRONMENT", "value": "production"},
        {"name": "MODEL_HOST", "value": "0.0.0.0"},
        {"name": "DATABASE_URL", "value": "postgresql://user:pass@rds-endpoint:5432/db"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/movielens-recsys",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8501/_stcore/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

### **Step 2: Deploy to ECS**
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

docker build -t movielens-recsys .
docker tag movielens-recsys:latest ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/movielens-recsys:latest
docker push ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/movielens-recsys:latest

# Create ECS service
aws ecs register-task-definition --cli-input-json file://task-definition.json
aws ecs create-service --cluster movielens-cluster --service-name movielens-service --task-definition movielens-recsys --desired-count 2
```

### **Step 3: Set up Load Balancer**
```bash
# Create Application Load Balancer
aws elbv2 create-load-balancer \
    --name movielens-alb \
    --subnets subnet-12345678 subnet-87654321 \
    --security-groups sg-12345678

# Create target group
aws elbv2 create-target-group \
    --name movielens-targets \
    --protocol HTTP \
    --port 8501 \
    --vpc-id vpc-12345678 \
    --health-check-path /_stcore/health
```

## 📊 **Production Monitoring Setup**

### **Step 1: Create Monitoring Configuration**
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "movielens_rules.yml"

scrape_configs:
  - job_name: 'movielens-app'
    static_configs:
      - targets: ['movielens-app:8501']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### **Step 2: Alerting Rules**
```yaml
# monitoring/movielens_rules.yml
groups:
  - name: movielens_alerts
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile latency is above 50ms"

      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.01
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is above 1%"

      - alert: ModelDrift
        expr: model_performance_score < 0.85
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Model performance degradation"
          description: "Model score dropped below threshold"
```

### **Step 3: Grafana Dashboard**
```json
{
  "dashboard": {
    "id": null,
    "title": "MovieLens Recommendation System",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Model Performance",
        "type": "stat",
        "targets": [
          {
            "expr": "model_performance_score",
            "legendFormat": "Current Score"
          }
        ]
      }
    ]
  }
}
```

## 🔒 **Security Best Practices**

### **1. Environment Security**
```bash
# Use secrets management
export JWT_SECRET=$(aws secretsmanager get-secret-value --secret-id prod/jwt-secret --query SecretString --output text)

# Network security
iptables -A INPUT -p tcp --dport 8501 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 8501 -j DROP
```

### **2. Application Security**
```python
# src/security/auth.py
import jwt
import secrets
from datetime import datetime, timedelta

class SecurityManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def generate_token(self, user_id: str) -> str:
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def validate_token(self, token: str) -> dict:
        try:
            return jwt.decode(token, self.secret_key, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            raise Exception("Token expired")
        except jwt.InvalidTokenError:
            raise Exception("Invalid token")
```

### **3. Data Security**
```python
# Encrypt sensitive data
from cryptography.fernet import Fernet

class DataEncryption:
    def __init__(self, key: bytes):
        self.cipher = Fernet(key)
    
    def encrypt_user_data(self, data: dict) -> str:
        return self.cipher.encrypt(json.dumps(data).encode()).decode()
    
    def decrypt_user_data(self, encrypted_data: str) -> dict:
        decrypted = self.cipher.decrypt(encrypted_data.encode())
        return json.loads(decrypted.decode())
```

## 📈 **Performance Optimization**

### **1. Caching Strategy**
```python
# Production caching configuration
REDIS_CONFIG = {
    'host': 'redis-cluster.xyz.cache.amazonaws.com',
    'port': 6379,
    'db': 0,
    'decode_responses': True,
    'max_connections': 100,
    'socket_timeout': 5,
    'socket_connect_timeout': 5,
    'health_check_interval': 30
}
```

### **2. Load Balancing**
```nginx
# nginx.conf for load balancing
upstream movielens_backend {
    least_conn;
    server app1:8501 weight=1 max_fails=3 fail_timeout=30s;
    server app2:8501 weight=1 max_fails=3 fail_timeout=30s;
    server app3:8501 weight=1 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name movielens.yourcompany.com;
    
    location / {
        proxy_pass http://movielens_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
}
```

## 🚨 **Incident Response Plan**

### **1. Monitoring Alerts**
```bash
# Set up alert notifications
curl -X POST https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK \
  -H 'Content-type: application/json' \
  --data '{"text":"🚨 MovieLens System Alert: High latency detected"}'
```

### **2. Rollback Procedure**
```bash
# Emergency rollback
aws ecs update-service --cluster movielens-cluster \
  --service movielens-service \
  --task-definition movielens-recsys:PREVIOUS_VERSION

# Or with Docker
docker-compose -f docker-compose.prod.yml down
docker tag movielens-recsys:v1.0.0 movielens-recsys:latest
docker-compose -f docker-compose.prod.yml up -d
```

### **3. Health Recovery**
```python
# Automated health recovery
def health_recovery():
    # Check system health
    health_status = run_health_check()
    
    if not health_status.all_healthy:
        # Restart unhealthy services
        restart_services(health_status.failed_services)
        
        # Clear caches
        clear_redis_cache()
        
        # Reload models
        reload_ml_models()
        
        # Notify team
        send_alert("System recovery initiated")
```

## 📋 **Post-Deployment Checklist**

### ✅ **Immediate Verification**
- [ ] Application accessible via load balancer
- [ ] Health check endpoints responding (/_stcore/health)
- [ ] Database connections established
- [ ] Redis cache working
- [ ] Monitoring dashboards showing data
- [ ] SSL certificates valid
- [ ] All environment variables set correctly

### ✅ **Performance Testing**
```bash
# Load testing with Apache Bench
ab -n 1000 -c 10 http://your-domain.com/

# API endpoint testing
curl -X POST http://your-domain.com/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 123, "n_recommendations": 10}'
```

### ✅ **Security Verification**
- [ ] No sensitive data in logs
- [ ] API rate limiting active
- [ ] HTTPS redirect working
- [ ] Firewall rules configured
- [ ] Secrets properly managed

## 🔄 **Continuous Deployment**

Your GitHub Actions workflows are already configured for:
- ✅ Automated testing on push
- ✅ Model validation with business thresholds
- ✅ Drift detection and retraining
- ✅ Safe rollback capabilities

## 📞 **Support & Monitoring**

### **Production URLs:**
- **Application**: `https://your-domain.com`
- **Monitoring**: `https://grafana.your-domain.com`
- **Metrics**: `https://prometheus.your-domain.com`
- **Health**: `https://your-domain.com/_stcore/health`

### **Emergency Contacts:**
- **Primary**: ml-team@yourcompany.com
- **Secondary**: devops@yourcompany.com
- **Slack**: #ml-alerts channel

---

🎉 **Your MovieLens Recommendation System is now production-ready with enterprise-grade deployment, monitoring, and security!**