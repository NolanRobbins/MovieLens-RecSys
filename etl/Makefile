# ETL Pipeline Makefile
# Production-ready commands for development, testing, and deployment

.PHONY: help test test-unit test-integration clean setup lint format check-deps run-etl run-simulation status docs

# Default target
help:
	@echo "🏗️  MovieLens ETL Pipeline Commands"
	@echo "=================================="
	@echo ""
	@echo "📊 Development:"
	@echo "  setup       - Set up development environment"
	@echo "  test        - Run all tests"
	@echo "  test-unit   - Run unit tests only"
	@echo "  test-integration - Run integration tests"
	@echo "  lint        - Run code linting"
	@echo "  format      - Format code"
	@echo "  check-deps  - Check dependencies"
	@echo ""
	@echo "🚀 ETL Operations:"
	@echo "  run-etl     - Run ETL pipeline (next day)"
	@echo "  run-simulation - Run 3-day simulation"
	@echo "  status      - Show pipeline status"
	@echo "  clean       - Clean generated files"
	@echo ""
	@echo "🧪 A/B Testing:"
	@echo "  run-ab-test - Run A/B testing comparison"
	@echo "  ab-report   - Show A/B testing report"
	@echo ""
	@echo "📚 Documentation:"
	@echo "  docs        - Generate documentation"

# Setup development environment
setup:
	@echo "🔧 Setting up ETL development environment..."
	@mkdir -p etl/logs
	@mkdir -p data/etl_batches/daily
	@mkdir -p data/etl_batches/cumulative
	@mkdir -p data/etl_metrics
	@mkdir -p etl/tests/__pycache__
	@echo "✅ Development environment ready"

# Run all tests
test: test-unit test-integration
	@echo "🎉 All ETL tests completed!"

# Run unit tests
test-unit:
	@echo "🧪 Running ETL unit tests..."
	@cd .. && python -m pytest etl/tests/test_etl_pipeline.py -v --tb=short
	@echo "✅ Unit tests completed"

# Run integration tests  
test-integration:
	@echo "🔗 Running ETL integration tests..."
	@cd .. && python etl/batch_etl_pipeline.py --simulate 2
	@cd .. && python etl/ab_testing/model_comparison.py --run
	@echo "✅ Integration tests completed"

# Code quality checks
lint:
	@echo "🔍 Running code linting..."
	@cd .. && python -m flake8 etl/ --max-line-length=100 --ignore=E203,W503 || echo "⚠️ Linting issues found"
	@echo "✅ Linting completed"

# Format code
format:
	@echo "🎨 Formatting code..."
	@cd .. && python -m black etl/ --line-length=100 || echo "⚠️ Black not available"
	@cd .. && python -m isort etl/ || echo "⚠️ isort not available"
	@echo "✅ Code formatting completed"

# Check dependencies
check-deps:
	@echo "📦 Checking dependencies..."
	@python -c "import pandas, numpy, yaml; print('✅ Core dependencies available')" || echo "❌ Missing core dependencies"
	@python -c "import unittest; print('✅ Testing framework available')" || echo "❌ Missing unittest"
	@echo "Dependencies check completed"

# ETL Operations
run-etl:
	@echo "🚀 Running ETL pipeline (next day)..."
	@cd .. && python etl/batch_etl_pipeline.py
	@echo "✅ ETL pipeline completed"

run-simulation:
	@echo "🎬 Running 3-day ETL simulation..."
	@cd .. && python etl/batch_etl_pipeline.py --simulate 3
	@echo "✅ Simulation completed"

status:
	@echo "📊 ETL Pipeline Status:"
	@echo "======================="
	@cd .. && python etl/batch_etl_pipeline.py --status

# A/B Testing
run-ab-test:
	@echo "🧪 Running A/B testing comparison..."
	@cd .. && python etl/ab_testing/model_comparison.py --run
	@echo "✅ A/B testing completed"

ab-report:
	@echo "📊 A/B Testing Report:"
	@echo "====================="
	@cd .. && python etl/ab_testing/model_comparison.py --status

# Cleanup
clean:
	@echo "🧹 Cleaning ETL pipeline files..."
	@rm -rf data/etl_batches/daily/*.csv
	@rm -rf data/etl_metrics/*.json
	@rm -rf etl/logs/*.log
	@rm -rf etl/tests/__pycache__
	@rm -rf etl/__pycache__
	@rm -rf etl/*/__pycache__
	@echo "✅ Cleanup completed"

# Generate documentation
docs:
	@echo "📚 Generating ETL documentation..."
	@echo "# ETL Pipeline API Documentation" > etl/docs/API.md
	@echo "" >> etl/docs/API.md
	@cd .. && python -c "from etl.batch_etl_pipeline import BatchETLPipeline; help(BatchETLPipeline)" >> etl/docs/API.md
	@echo "✅ Documentation generated in etl/docs/"

# Development shortcuts
quick-test: setup test-unit
	@echo "🚀 Quick test cycle completed"

full-cycle: setup test run-simulation run-ab-test
	@echo "🎯 Full development cycle completed"

# GitHub Actions simulation
ci-test:
	@echo "🤖 Simulating GitHub Actions CI..."
	@make setup
	@make test-unit
	@make run-simulation
	@make run-ab-test
	@echo "✅ CI simulation completed"

# Production readiness check
prod-check:
	@echo "🏭 Production readiness check..."
	@make check-deps
	@make test
	@make lint
	@cd .. && python etl/batch_etl_pipeline.py --status
	@echo "✅ Production readiness check completed"

# Monitoring
logs:
	@echo "📋 Recent ETL logs:"
	@echo "=================="
	@ls -la etl/logs/ | head -10

metrics:
	@echo "📊 ETL Metrics:"
	@echo "=============="
	@ls -la data/etl_metrics/ | head -10

# Emergency commands
reset-pipeline:
	@echo "🚨 Resetting ETL pipeline state..."
	@rm -f data/etl_batches/processing_tracker.json
	@echo "⚠️ Pipeline reset - will start from day 1"

force-day:
	@echo "🎯 Processing specific day: $(DAY)"
	@cd .. && python etl/batch_etl_pipeline.py --day $(DAY)

# Performance testing
perf-test:
	@echo "⚡ Running performance tests..."
	@cd .. && time python etl/batch_etl_pipeline.py --simulate 5
	@echo "✅ Performance test completed"

# Development workflow
dev-workflow: setup test-unit run-etl status
	@echo "🔄 Development workflow completed"

# Streamlit integration check
streamlit-check:
	@echo "🖥️  Checking Streamlit integration..."
	@cd .. && python -c "from etl.ab_testing.model_comparison import ETL_AB_TestingFramework; print('✅ Streamlit integration ready')"

# Show current pipeline progress
progress:
	@echo "📈 ETL Pipeline Progress:"
	@echo "========================"
	@cd .. && python -c "
import json, os
if os.path.exists('data/etl_batches/processing_tracker.json'):
    with open('data/etl_batches/processing_tracker.json') as f:
        tracker = json.load(f)
    print(f'Day: {tracker.get(\"current_day\", 0)}/20')
    print(f'Progress: {tracker.get(\"completion_percentage\", 0):.1f}%')
    print(f'Records: {tracker.get(\"processed_records\", 0):,}')
else:
    print('No progress data available')
"

# Install dev dependencies (if needed)
install-dev-deps:
	@echo "📦 Installing development dependencies..."
	@pip install pytest flake8 black isort || echo "⚠️ Some dev dependencies may need manual installation"
