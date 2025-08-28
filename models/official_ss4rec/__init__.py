"""
Official SS4Rec Implementation Package

This package contains the official SS4Rec implementation using:
- RecBole 1.0 framework for standard evaluation
- Official mamba-ssm==2.2.2 for Relation-Aware SSM
- Official s5-pytorch==0.2.1 for Time-Aware SSM

Usage:
    from models.official_ss4rec import SS4RecOfficial, create_ss4rec_config
"""

from .ss4rec_official import SS4RecOfficial, create_ss4rec_config

__all__ = ['SS4RecOfficial', 'create_ss4rec_config']