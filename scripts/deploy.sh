#!/bin/bash

# MovieLens Recommendation System Deployment Script
# Comprehensive deployment for production environment

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="MovieLens RecSys"
DOCKER_COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to check if data exists
check_data() {
    print_status "Checking data availability..."
    
    if [ ! -d "data/raw/ml-32m" ]; then
        print_warning "MovieLens dataset not found. The system will run in demo mode."
    else
        print_success "MovieLens dataset found"
    fi
    
    if [ ! -d "data/processed" ]; then
        print_status "Processed data not found. Running ETL pipeline..."
        run_etl_pipeline
    else
        print_success "Processed data found"
    fi
}

# Function to run ETL pipeline
run_etl_pipeline() {
    print_status "Running ETL pipeline..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment and install dependencies
    source venv/bin/activate
    pip install -r requirements.txt
    
    # Run ETL pipeline
    if [ -d "ml-32m" ]; then
        python etl_pipeline.py --data_source ./ml-32m --output ./processed_data
        print_success "ETL pipeline completed successfully"
    else
        print_warning "No source data found. Skipping ETL pipeline."
    fi
    
    deactivate
}

# Function to create environment file
create_env_file() {
    print_status "Creating environment configuration..."
    
    if [ ! -f "$ENV_FILE" ]; then
        cat > "$ENV_FILE" << EOF
# Environment Configuration
ENV=production
PROJECT_NAME=movielens-recsys

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Demo Configuration
DEMO_HOST=0.0.0.0
DEMO_PORT=8501

# Model Configuration
MODEL_PATH=/app/models/hybrid_vae_best.pt
DATA_PATH=/app/processed_data

# Monitoring
LOG_LEVEL=INFO
ENABLE_METRICS=true

# Cache Configuration (if using Redis)
REDIS_URL=redis://redis:6379/0
CACHE_TTL=3600

# Security
SECRET_KEY=your-secret-key-change-in-production
ALLOWED_ORIGINS=*
EOF
        print_success "Environment file created"
    else
        print_status "Environment file already exists"
    fi
}

# Function to build Docker images
build_images() {
    print_status "Building Docker images..."
    
    docker-compose build --no-cache
    print_success "Docker images built successfully"
}

# Function to deploy services
deploy_services() {
    print_status "Deploying services..."
    
    # Stop any existing services
    docker-compose down --remove-orphans
    
    # Start core services
    docker-compose up -d api demo
    
    print_status "Waiting for services to be healthy..."
    sleep 10
    
    # Check if services are healthy
    if docker-compose ps | grep -q "Up (healthy)"; then
        print_success "Core services deployed successfully"
    else
        print_error "Service deployment failed. Check logs with: docker-compose logs"
        exit 1
    fi
}

# Function to deploy with monitoring
deploy_with_monitoring() {
    print_status "Deploying with monitoring stack..."
    
    # Create monitoring directory structure
    mkdir -p monitoring/{prometheus,grafana/{dashboards,datasources}}
    
    # Create basic Prometheus config
    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'movielens-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s
EOF
    
    # Deploy with monitoring profile
    docker-compose --profile monitoring up -d
    
    print_success "Services with monitoring deployed"
}

# Function to run health checks
health_check() {
    print_status "Running health checks..."
    
    # Check API health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "API service is healthy"
    else
        print_error "API service health check failed"
        return 1
    fi
    
    # Check Demo interface
    if curl -f http://localhost:8501 > /dev/null 2>&1; then
        print_success "Demo interface is accessible"
    else
        print_warning "Demo interface may not be ready yet (this is normal during startup)"
    fi
    
    return 0
}

# Function to show deployment info
show_deployment_info() {
    echo ""
    echo "=========================================="
    echo "🎬 $PROJECT_NAME Deployment Complete! 🎬"
    echo "=========================================="
    echo ""
    echo "📊 Services Available:"
    echo "  • API Service:      http://localhost:8000"
    echo "  • API Health:       http://localhost:8000/health"
    echo "  • API Docs:         http://localhost:8000/docs"
    echo "  • Demo Interface:   http://localhost:8501"
    echo ""
    
    if docker-compose ps | grep -q prometheus; then
        echo "📈 Monitoring:"
        echo "  • Prometheus:       http://localhost:9090"
        echo "  • Grafana:          http://localhost:3000 (admin/admin)"
        echo ""
    fi
    
    echo "🔧 Management Commands:"
    echo "  • View logs:        docker-compose logs -f"
    echo "  • Stop services:    docker-compose down"
    echo "  • Restart:          docker-compose restart"
    echo "  • Run ETL:          docker-compose --profile etl up etl"
    echo ""
    
    echo "📁 Important Files:"
    echo "  • Configuration:    $ENV_FILE"
    echo "  • Docker Compose:   $DOCKER_COMPOSE_FILE"
    echo "  • Processed Data:   ./processed_data/"
    echo ""
    
    print_success "Deployment information displayed above"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --quick          Quick deployment (skip some checks)"
    echo "  --monitoring     Deploy with monitoring stack"
    echo "  --etl-only       Run only ETL pipeline"
    echo "  --health-check   Run health checks only"
    echo "  --clean          Clean up all containers and volumes"
    echo "  --help           Show this help message"
    echo ""
}

# Function to clean up
cleanup() {
    print_status "Cleaning up containers and volumes..."
    
    docker-compose down --volumes --remove-orphans
    docker system prune -f
    
    print_success "Cleanup completed"
}

# Main deployment function
main() {
    local QUICK_MODE=false
    local MONITORING_MODE=false
    local ETL_ONLY=false
    local HEALTH_CHECK_ONLY=false
    local CLEAN_MODE=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick)
                QUICK_MODE=true
                shift
                ;;
            --monitoring)
                MONITORING_MODE=true
                shift
                ;;
            --etl-only)
                ETL_ONLY=true
                shift
                ;;
            --health-check)
                HEALTH_CHECK_ONLY=true
                shift
                ;;
            --clean)
                CLEAN_MODE=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Handle specific modes
    if [ "$CLEAN_MODE" = true ]; then
        cleanup
        exit 0
    fi
    
    if [ "$HEALTH_CHECK_ONLY" = true ]; then
        health_check
        exit $?
    fi
    
    if [ "$ETL_ONLY" = true ]; then
        run_etl_pipeline
        exit 0
    fi
    
    # Main deployment flow
    echo "🚀 Starting $PROJECT_NAME Deployment..."
    echo ""
    
    if [ "$QUICK_MODE" = false ]; then
        check_prerequisites
        check_data
    fi
    
    create_env_file
    build_images
    
    if [ "$MONITORING_MODE" = true ]; then
        deploy_with_monitoring
    else
        deploy_services
    fi
    
    # Wait a bit for services to fully start
    print_status "Waiting for services to stabilize..."
    sleep 15
    
    if health_check; then
        show_deployment_info
        print_success "🎉 Deployment completed successfully!"
    else
        print_error "Deployment completed with warnings. Check service logs."
        echo "Run 'docker-compose logs' to see detailed logs."
        exit 1
    fi
}

# Run main function with all arguments
main "$@"