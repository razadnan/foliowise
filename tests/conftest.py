"""
tests/conftest.py
=================
Pytest configuration and global fixtures.
Ensures repository root is always in sys.path regardless of execution directory.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
