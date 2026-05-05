"""
Marg ERP Data Extraction Pipeline
=================================
A completely in-house, privacy-first ETL pipeline for parsing 
chaotic Marg ERP pharmaceutical reports into structured CSV/JSON data.
"""

from .main import process_document

__all__ = ["process_document"]