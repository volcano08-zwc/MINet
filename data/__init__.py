"""BCI Competition IV 2a input pipeline."""

from .bci2a import BCI2aDataset, BranchNormalizer, read_session, stratified_split

__all__ = ["BCI2aDataset", "BranchNormalizer", "read_session", "stratified_split"]
