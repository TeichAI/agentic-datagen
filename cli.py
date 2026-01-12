#!/usr/bin/env python3
"""
Agentic Dataset Generator CLI

Usage:
    python -m agentic_datagen.cli -c config.yaml
"""

import sys
from pathlib import Path

# Add parent directory to path to access scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_datagen.generator import main

if __name__ == "__main__":
    main()
