"""Lets the CLI run as `python -m vsearch ...`."""
import sys

from .cli import main

sys.exit(main())
