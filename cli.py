#!/usr/bin/env python3
"""
Agentic Dataset Generator CLI

Usage:
    python -m agentic_datagen.cli -c config.yaml
"""

import sys
from pathlib import Path

# Add repo root so local modules can be imported directly
sys.path.insert(0, str(Path(__file__).parent))

from generator import main

if __name__ == "__main__":
    main()
