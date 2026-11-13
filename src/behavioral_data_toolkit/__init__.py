"""Open Behavioral Data Toolkit.

A reproducible toolkit for behavioral research: data loading and validation,
questionnaire scoring, descriptive statistics, correlation analysis and
reporting. See the documentation in the ``docs/`` directory of the repository.

The toolkit performs descriptive statistics only. It provides no diagnoses and no
clinical conclusions.
"""

from ._version import __version__
from .correlation import CorrelationMatrix, CorrelationResult, correlation_matrix, pearson, spearman
from .exceptions import (
    BehavioralDataToolkitError,
    ColumnError,
    DataLoadError,
    InputValidationError,
    ScoringError,
    StatisticsError,
)
from .loader import DataStructure, inspect_dataframe, load_csv, validate_columns
from .plotting import plot_correlation_heatmap, plot_distribution
from .reporting import AnalysisReport, collect_environment_metadata, dataframe_fingerprint
from .scoring import (
    ScaleDefinition,
    composite_score,
    reverse_score,
    score_questionnaire,
    score_questionnaires,
)
from .stats import (
    CronbachAlphaResult,
    DistributionSummary,
    cronbach_alpha,
    distribution_summary,
    mean,
    median,
    std,
)
from .validator import (
    DataQualitySummary,
    ValueRule,
    data_quality_summary,
    find_duplicates,
    find_invalid_values,
    missing_summary,
)

__all__ = [
    "AnalysisReport",
    "BehavioralDataToolkitError",
    "ColumnError",
    "CorrelationMatrix",
    "CorrelationResult",
    "CronbachAlphaResult",
    "DataLoadError",
    "DataQualitySummary",
    "DataStructure",
    "DistributionSummary",
    "InputValidationError",
    "ScaleDefinition",
    "ScoringError",
    "StatisticsError",
    "ValueRule",
    "__version__",
    "collect_environment_metadata",
    "composite_score",
    "correlation_matrix",
    "cronbach_alpha",
    "data_quality_summary",
    "dataframe_fingerprint",
    "distribution_summary",
    "find_duplicates",
    "find_invalid_values",
    "inspect_dataframe",
    "load_csv",
    "mean",
    "median",
    "missing_summary",
    "pearson",
    "plot_correlation_heatmap",
    "plot_distribution",
    "reverse_score",
    "score_questionnaire",
    "score_questionnaires",
    "spearman",
    "std",
    "validate_columns",
]
