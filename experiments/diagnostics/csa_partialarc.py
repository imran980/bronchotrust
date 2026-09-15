"""Moved to pipeline/csa_partialarc.py; this shim keeps old imports working."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "pipeline"))
from csa_partialarc import *  # noqa
