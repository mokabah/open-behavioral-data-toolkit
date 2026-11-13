"""Exception hierarchy for Open Behavioral Data Toolkit."""

from __future__ import annotations


class BehavioralDataToolkitError(Exception):
    """Base class for all errors raised deliberately by this package."""


class DataLoadError(BehavioralDataToolkitError):
    """A data file could not be read or parsed."""


class InputValidationError(BehavioralDataToolkitError, ValueError):
    """A function received an argument that is invalid or of the wrong kind."""


class ColumnError(InputValidationError):
    """Required columns are missing, or columns have an unsuitable type."""


class ScoringError(InputValidationError):
    """Questionnaire scoring cannot proceed (e.g. responses outside the scale)."""


class StatisticsError(BehavioralDataToolkitError):
    """A statistic is mathematically undefined for the supplied data."""
