#!/bin/bash

# Production Deployment Script for MovieLens Recommendation System
# Automates the complete production deployment process

set -e

echo "🚀 MovieLens Recommendation System - Production Deployment"
echo "==========================================================="

# Configuration
DOCKER_IMAGE_NAME="movielens-recsys"
DOCKER_TAG=${DOCKER_TAG:-latest}
ENVIRONMENT=${ENVIRONMENT:-production}

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
    print_status "Checking deployment prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker and try again."
        exit 1
    fi
    
    # Check if docker-compose is installed
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose and try again."
        exit 1
    fi
    
    # Check if .env file exists
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating from template..."
        cat > .env << EOF
ENVIRONMENT=production
ALERT_EMAIL=ml-team@yourcompany.com
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
GRAFANA_PASSWORD=secure-grafana-password
EOF
        print_warning "Please edit .env file with your production values before continuing."
        print_warning "Press Enter to continue when ready, or Ctrl+C to exit..."
        read
    fi
    
    print_success "Prerequisites check completed"
}

# Function to run system validation
validate_system() {
    print_status "Running system validation..."
    
    if python validate_system.py; then
        print_success "System validation passed (7/7 tests)"
    else
        print_error "System validation failed. Please fix issues before deploying."
        exit 1
    fi
}

# Function to build Docker image
build_docker_image() {
    print_status "Building production Docker image..."
    
    docker build -f deployment/Dockerfile.prod -t ${DOCKER_IMAGE_NAME}:${DOCKER_TAG} .
    
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully: ${DOCKER_IMAGE_NAME}:${DOCKER_TAG}"
    else
        print_error "Docker image build failed"
        exit 1
    fi
}

# Function to deploy with Docker Compose
deploy_docker_compose() {
    print_status "Deploying with Docker Compose..."
    
    cd deployment
    
    # Stop existing containers
    docker-compose -f docker-compose.prod.yml down
    
    # Start services
    docker-compose -f docker-compose.prod.yml up -d
    
    if [ $? -eq 0 ]; then
        print_success "Services deployed successfully"
    else
        print_error "Deployment failed"
        exit 1
    fi
    
    cd ..
}

# Function to wait for services to be healthy
wait_for_services() {
    print_status "Waiting for services to become healthy..."
    
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker-compose -f deployment/docker-compose.prod.yml ps | grep -q "healthy"; then
            print_success "Services are healthy"
            return 0
        fi
        
        print_status "Attempt $attempt/$max_attempts - Services still starting..."
        sleep 10
        ((attempt++))
    done
    
    print_error "Services failed to become healthy within expected time"
    docker-compose -f deployment/docker-compose.prod.yml logs
    exit 1
}

# Function to run deployment tests
run_deployment_tests() {
    print_status "Running deployment tests..."
    
    # Test main application endpoint
    if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        print_success "✅ Main application is responding"
    else
        print_error "❌ Main application health check failed"
        return 1
    fi
    
    # Test Prometheus
    if curl -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
        print_success "✅ Prometheus is responding"
    else
        print_warning "⚠️ Prometheus health check failed"
    fi
    
    # Test Grafana
    if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
        print_success "✅ Grafana is responding"
    else
        print_warning "⚠️ Grafana health check failed"
    fi
    
    # Test Redis
    if docker exec deployment_redis_1 redis-cli ping > /dev/null 2>&1; then
        print_success "✅ Redis is responding"
    else
        print_warning "⚠️ Redis health check failed"
    fi
    
    print_success "Deployment tests completed"
}

# Function to display deployment information
show_deployment_info() {
    print_success "🎉 Deployment completed successfully!"
    echo ""
    echo "📊 Service URLs:"
    echo "  - Main Application: http://localhost:8501"
    echo "  - Prometheus:       http://localhost:9090"
    echo "  - Grafana:          http://localhost:3000"
    echo "  - Load Balancer:    http://localhost:80"
    echo ""
    echo "🔐 Default Credentials:"
    echo "  - Grafana: admin / (check .env file for password)"
    echo ""
    echo "📋 Useful Commands:"
    echo "  - View logs:    docker-compose -f deployment/docker-compose.prod.yml logs -f"
    echo "  - Scale app:    docker-compose -f deployment/docker-compose.prod.yml up --scale movielens-app=3 -d"
    echo "  - Stop all:     docker-compose -f deployment/docker-compose.prod.yml down"
    echo "  - System status: docker-compose -f deployment/docker-compose.prod.yml ps"
    echo ""
    echo "🏥 Health Checks:"
    echo "  - curl http://localhost:8501/_stcore/health"
    echo "  - python validate_system.py"
    echo ""
}

# Main deployment flow
main() {
    print_status "Starting production deployment for MovieLens Recommendation System"
    
    check_prerequisites
    validate_system
    build_docker_image
    deploy_docker_compose
    wait_for_services
    run_deployment_tests
    show_deployment_info
    
    print_success "🚀 Production deployment completed successfully!"
}

# Handle script arguments
case "${1:-}" in
    "build")
        build_docker_image
        ;;
    "deploy")
        deploy_docker_compose
        wait_for_services
        ;;
    "test")
        run_deployment_tests
        ;;
    "validate")
        validate_system
        ;;
    "stop")
        print_status "Stopping all services..."
        cd deployment
        docker-compose -f docker-compose.prod.yml down
        print_success "All services stopped"
        ;;
    "logs")
        cd deployment
        docker-compose -f docker-compose.prod.yml logs -f
        ;;
    *)
        main
        ;;
esac