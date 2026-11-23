"""
tests/conftest.py
=================
Root test configuration and early binary module stabilization for Windows Python 3.12.
Ensures C-extensions (optree, torch) are cleanly loaded before thread pool initialization.
"""

import optree
import torch

# Disable dynamo JIT hooks during unit tests on Windows
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
except Exception:
    pass
