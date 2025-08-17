# Claude.md: Rules for AI-Assisted Software Engineering

This document outlines the rules and best practices for our collaboration. The primary goals are to maximize workflow productivity, ensure the generation of correct and useful code, adhere to proper software engineering principles, and maintain conciseness to limit token usage.

---

## 1. Core Principles & Philosophy

- **Assume Seniority**: Act as a senior ML engineer. The code should be production-ready, robust, and well-documented.
- **Conciseness is Key**: Be direct and to the point. Avoid conversational filler. Focus on delivering high-quality output efficiently.
- **Iterative Development**: We will work iteratively. Start with a simple, working version and progressively add complexity. Refactoring is expected and encouraged.
- **Question Ambiguity**: If a prompt is unclear or lacks detail, ask for clarification before generating code. It's better to ask than to generate incorrect output.
- **Fail Fast**: Design systems to fail early and loudly during development rather than silently producing incorrect results in production.

---

## 2. Environment & Dependency Management

- **Virtual Environments with `uv`**: All projects must start by creating a virtual environment. Use `uv` for its speed.
  - Create env: `uv venv`
  - Activate: `source .venv/bin/activate`
- **Dependency Management**: Use `uv` to manage dependencies. All projects must have a `requirements.txt` file.
  - Install packages: `uv pip install -r requirements.txt`
  - Freeze dependencies: `uv pip freeze > requirements.txt`
- **Reproducibility**: The `requirements.txt` file ensures that the project is reproducible in any environment. Always keep it up to date.
- **Version Pinning**: Pin exact versions for production (`==`), use flexible versioning for libraries (`>=`, `<`).

---

## 3. Code Generation & Quality

- **Modularity**: Break down complex problems into smaller, single-purpose functions or classes. Follow the "Do Not Repeat Yourself" (DRY) principle.
- **Readability**:
    - **Formatting**: All Python code must be formatted with **Black**. Use `isort` to sort imports.
    - **Naming**: Adhere to standard Python naming conventions (PEP 8).
- **Static Type Checking**: All functions, methods, and variables must use type annotations. This is mandatory. See section 3.1 for details.
- **Performance**:
    - Use efficient, built-in data structures and functions (e.g., `collections`, `itertools`).
    - Prefer vectorized operations with **NumPy/Pandas** over Python loops.
    - For performance-critical code, use profilers (`cProfile`, `line_profiler`) to identify bottlenecks before optimizing.
- **Robustness**:
    - **Error Handling**: Use `try...except` blocks to handle potential exceptions gracefully. Provide informative error messages.
    - **Logging**: Use the `logging` module instead of `print()` for debugging and tracking events. Configure it to log to a file with timestamps.
    - **Assertions**: Use assertions liberally during development to validate assumptions about data shapes, types, and ranges.

### 3.1. Static Type Checking with Type Annotations

To catch type-related bugs before runtime and improve code clarity, we will use type annotations (also known as type hints) for all functions and class methods.

-   **Syntax**: Annotate all function arguments and return values using standard Python type hints.
    -   *Example*:
        ```python
        from typing import List, Dict, Optional
        import pandas as pd
        import numpy as np
        from numpy.typing import NDArray

        def process_data(
            user_ids: List[int], 
            is_active: bool,
            threshold: Optional[float] = None
        ) -> Dict[str, int]:
            # function logic...
            return {"processed_ids": len(user_ids)}
        
        def transform_features(
            df: pd.DataFrame,
            feature_cols: List[str]
        ) -> NDArray[np.float64]:
            # transformation logic...
            return df[feature_cols].values
        ```
-   **Enforcement**: We will use **mypy** as our static type checker to validate these annotations. Code must pass `mypy` checks to be considered complete.
-   **Automation**: `mypy` will be integrated into our pre-commit hooks to ensure that no code with type errors is committed to the repository.

---

## 4. Testing & Validation

### 4.1. Testing Strategy

- **Test Coverage Targets**: 
    - Aim for 80%+ code coverage for business logic
    - 100% coverage for data transformation and model inference code
    - Use `pytest-cov` to track coverage: `pytest --cov=src --cov-report=html`

- **Testing Pyramid**:
    1. **Unit Tests** (70% of tests):
        - Test individual functions in isolation
        - Mock external dependencies
        - Run in milliseconds
        - File naming: `test_unit_*.py`
    
    2. **Integration Tests** (20% of tests):
        - Test component interactions
        - Use real databases/APIs in test mode
        - File naming: `test_integration_*.py`
    
    3. **End-to-End Tests** (10% of tests):
        - Test complete workflows
        - Validate model training pipelines end-to-end
        - File naming: `test_e2e_*.py`

- **Test Organization**:
    ```python
    # Example test structure with fixtures
    import pytest
    from typing import Generator
    import pandas as pd
    
    @pytest.fixture
    def sample_data() -> pd.DataFrame:
        """Fixture providing test data."""
        return pd.DataFrame({
            'feature_1': [1, 2, 3],
            'feature_2': [4, 5, 6],
            'target': [0, 1, 0]
        })
    
    def test_data_transformation(sample_data: pd.DataFrame) -> None:
        """Test that data transformation preserves shape."""
        # Arrange
        expected_shape = (3, 2)
        
        # Act
        result = transform_features(sample_data, ['feature_1', 'feature_2'])
        
        # Assert
        assert result.shape == expected_shape
        assert result.dtype == np.float64
    ```

- **Data Validation**: When working with data (especially for APIs or ML models), use **Pandera** or **Pydantic** to define a schema and validate dataframes or data models. This catches data quality issues early.

- **Property-Based Testing**: For critical functions, use **Hypothesis** to generate test cases automatically:
    ```python
    from hypothesis import given, strategies as st
    
    @given(st.lists(st.integers(), min_size=1))
    def test_process_never_returns_negative(user_ids: List[int]) -> None:
        result = process_data(user_ids, True)
        assert result["processed_ids"] >= 0
    ```

- **CI/CD Automation**:
    - **Pre-commit Hooks**: Set up a `.pre-commit-config.yaml` file to run `black`, `isort`, `ruff`, and `mypy` automatically before each commit. This enforces quality standards with zero effort.
    - **GitHub Actions**: For larger projects, create a simple GitHub Actions workflow (`.github/workflows/ci.yml`) to automatically run tests on every push to `main`.

---

## 5. Project Structure & Documentation

- **Standard Project Layout**: Use a standardized project structure.
```
project_name/
├── .venv/
├── .github/workflows/ci.yml
├── configs/                # Configuration files (YAML/JSON)
│   ├── model_config.yaml
│   └── training_config.yaml
├── data/                   # Data directory (gitignored)
│   ├── raw/
│   ├── processed/
│   └── features/
├── notebooks/              # Experimental notebooks
│   └── exploratory/        # EDA and prototyping
├── src/                    # Main source code
│   ├── __init__.py
│   ├── data/              # Data loading and processing
│   ├── features/          # Feature engineering
│   ├── models/            # Model definitions and training
│   ├── evaluation/        # Model evaluation and metrics
│   └── utils/             # Utility functions
├── tests/                  # All tests
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── scripts/                # Standalone scripts
│   ├── train_model.py
│   └── evaluate_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── Makefile                # Common commands
├── README.md
├── requirements.txt
└── requirements-dev.txt    # Development dependencies
```

- **Documentation**:
  - **Docstrings**: All modules, classes, and functions must have Google-style docstrings explaining their purpose, arguments, and return values.
  - **README.md**: Every project must have a `README.md` that explains what the project does, how to set it up, and how to run it.
  - **Comments**: Use comments only to explain *why* something is done a certain way, not *what* the code is doing. The code should be self-explanatory.
  - **Decision Log**: Maintain a `docs/decisions.md` file documenting key architectural and algorithmic decisions with rationale.

---

## 6. ML Engineering Best Practices

### 6.1. Data Pipeline

- **Data Versioning**: Use DVC or similar tools to version datasets. Never commit data to git.
- **Data Quality Checks**:
    ```python
    # Always validate data at pipeline boundaries
    def validate_input_data(df: pd.DataFrame) -> None:
        assert not df.empty, "Input dataframe is empty"
        assert df.isnull().sum().sum() < len(df) * 0.1, "Too many nulls (>10%)"
        assert len(df) >= MIN_SAMPLES, f"Insufficient samples: {len(df)}"
    ```
- **Feature Store**: For production systems, implement a simple feature store pattern to ensure training/serving consistency.

### 6.2. Model Development

- **Experiment Tracking**: 
    - Use Weights & Biases for all experiments
    - Log hyperparameters, metrics, and artifacts for every run
    - Tag experiments with git commit hash for reproducibility

- **Model Validation**:
    ```python
    # Always validate model outputs
    def validate_predictions(preds: NDArray[np.float64]) -> None:
        assert np.all(np.isfinite(preds)), "Model produced NaN/Inf"
        assert preds.min() >= 0 and preds.max() <= 1, "Probabilities out of range"
    ```

- **Cross-Validation**: Always use proper cross-validation:
    - Time series: Use TimeSeriesSplit
    - Imbalanced data: Use StratifiedKFold
    - Standard: Use KFold with shuffle=True

### 6.3. Model Deployment

- **Model Registry**: Use a model registry (MLflow, BentoML) to manage model versions.
- **Model Serialization**: 
    - Never use `pickle` for production
    - Prefer: ONNX (cross-platform), SavedModel (TensorFlow), TorchScript (PyTorch)
    - For sklearn: Use `skops` or convert to ONNX

- **Monitoring & Observability**:
    - Log prediction latencies, throughput, and error rates
    - Implement data drift detection (use libraries like `evidently` or `alibi-detect`)
    - Set up alerts for model performance degradation

- **A/B Testing**: Always deploy new models alongside existing ones initially:
    ```python
    # Shadow mode deployment pattern
    def predict(features: pd.DataFrame) -> Dict[str, Any]:
        pred_v1 = model_v1.predict(features)  # Current model
        pred_v2 = model_v2.predict(features)  # New model (shadow)
        
        # Log both predictions for comparison
        log_predictions({"v1": pred_v1, "v2": pred_v2})
        
        # Return current model's prediction
        return {"prediction": pred_v1, "model_version": "v1"}
    ```

---

## 7. API & System Design

- **APIs**: When building APIs, use **FastAPI** for its performance, automatic documentation (Swagger UI), and Pydantic integration. Keep API logic separate from business logic.

- **Batch vs Real-time**:
    - Default to batch processing when possible (simpler, more efficient)
    - Use real-time only when latency requirements demand it (<100ms)
    - Consider hybrid: Real-time serving with batch feature computation

- **Caching Strategy**:
    - Cache expensive computations (feature engineering, model predictions)
    - Use Redis for distributed caching, `functools.lru_cache` for local

---

## 8. Performance & Scalability

- **Profiling First**: Never optimize without profiling first. Use:
    - `cProfile` for general profiling
    - `memory_profiler` for memory usage
    - `py-spy` for production profiling

- **Vectorization**: Always prefer vectorized operations:
    ```python
    # Bad: Loop-based processing
    results = []
    for item in data:
        results.append(process(item))
    
    # Good: Vectorized processing
    results = np.vectorize(process)(data)
    # Or better: Native NumPy/Pandas operations
    results = data.apply(process)
    ```

- **Parallelization**: Use `joblib` or `multiprocessing` for CPU-bound tasks:
    ```python
    from joblib import Parallel, delayed
    
    results = Parallel(n_jobs=-1)(
        delayed(process_chunk)(chunk) 
        for chunk in data_chunks
    )
    ```

---

## 9. Common Pitfalls to Avoid

1. **Data Leakage**: Always split data BEFORE any preprocessing
2. **Random Seeds**: Set seeds everywhere (numpy, random, torch, etc.) for reproducibility
3. **Silent Failures**: Never use bare `except:` clauses
4. **Memory Leaks**: Watch for growing lists/dicts in loops, use generators when possible
5. **Mutable Defaults**: Never use mutable default arguments in functions

---

## 10. Makefile Commands

Include a Makefile for common operations:
```makefile
.PHONY: install test lint format clean

install:
	uv pip install -r requirements.txt
	uv pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	mypy src/
	ruff check src/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov/

train:
	python scripts/train_model.py --config configs/training_config.yaml

evaluate:
	python scripts/evaluate_model.py --model-path models/latest.onnx
```