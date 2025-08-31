"""
Official SS4Rec Implementation Package

This package contains the official SS4Rec implementation using:
- RecBole 1.0 framework for standard evaluation
- Official mamba-ssm==2.2.2 for Relation-Aware SSM
- Official s5-pytorch==0.2.1 for Time-Aware SSM

Usage:
    from models.official_ss4rec import SS4RecOfficial, create_ss4rec_config
"""

# Defer imports to avoid RecBole dependency issues during import
def _lazy_import():
    """Lazy import to avoid issues when RecBole isn't available"""
    from .ss4rec_official import SS4RecOfficial, create_ss4rec_config
    return SS4RecOfficial, create_ss4rec_config

# Only import when actually accessed
def __getattr__(name):
    if name in ['SS4RecOfficial', 'create_ss4rec_config']:
        SS4RecOfficial, create_ss4rec_config = _lazy_import()
        globals()['SS4RecOfficial'] = SS4RecOfficial
        globals()['create_ss4rec_config'] = create_ss4rec_config
        return globals()[name]
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = ['SS4RecOfficial', 'create_ss4rec_config']