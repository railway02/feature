"""Reference-faithful local DSA registration Stage A-C utilities.

This package intentionally contains no cohort registration runner.  The current scope is
data-contract preflight, independent-local-crop geometry audit, and synthetic validation.
"""

from .preprocessing_adapter import PairRecord, PhaseRecord, load_local_reference_pairs

__all__ = ["PairRecord", "PhaseRecord", "load_local_reference_pairs"]
