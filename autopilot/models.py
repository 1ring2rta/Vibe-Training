from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TrainingType(str, Enum):
    SFT = "sft"
    DPO = "dpo"
    REWARD_MODEL = "reward_model"
    KTO = "kto"
    RL = "rl"
    RLVR = "rlvr"
    CONTINUED_PRETRAINING = "continued_pretraining"
    UNKNOWN = "unknown"


class RiskSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class RiskFlag:
    """A non-security usability flag.

    The project currently uses this for data usefulness/convertibility issues,
    not safety screening.
    """

    name: str
    severity: RiskSeverity
    description: str
    count: int = 0
    examples: list[str] = field(default_factory=list)


@dataclass
class DatasetSearchResult:
    dataset_id: str
    author: str | None = None
    downloads: int | None = None
    likes: int | None = None
    tags: list[str] = field(default_factory=list)
    license: str | None = None
    last_modified: str | None = None
    gated: bool | str | None = None
    private: bool | None = None
    source: str = "hf_api"


@dataclass
class DatasetFileInfo:
    path: str
    size: int | None = None
    blob_id: str | None = None
    lfs: bool | None = None


@dataclass
class DatasetWebOverview:
    dataset_id: str
    hub_url: str
    card_excerpt: str = ""
    files: list[DatasetFileInfo] = field(default_factory=list)
    viewer_status: dict[str, Any] = field(default_factory=dict)
    configs_splits: list[dict[str, Any]] = field(default_factory=list)
    selected_config: str | None = None
    selected_split: str | None = None
    example_rows: list[dict[str, Any]] = field(default_factory=list)
    browse_error: str | None = None


@dataclass
class DatasetInspection:
    dataset_id: str
    config_name: str | None
    split: str | None
    columns: list[str]
    features: dict[str, str]
    sample_rows: list[dict[str, Any]]
    metadata: DatasetSearchResult
    load_error: str | None = None
    web_overview: DatasetWebOverview | None = None


@dataclass
class DatasetClassification:
    format_type: str
    recommended_training: list[TrainingType]
    confidence: float
    reasons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)


@dataclass
class RiskAssessment:
    """Quality/usability assessment, not safety assessment."""

    risk_score: float
    quality_score: float
    flags: list[RiskFlag] = field(default_factory=list)
    sample_count: int = 0
    duplicate_rate: float = 0.0
    avg_text_length: float = 0.0
    chinese_char_ratio: float = 0.0
    llm_review: dict[str, Any] | None = None


@dataclass
class DatasetScore:
    suitability_score: float
    components: dict[str, float]
    reasons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)


@dataclass
class PromptExample:
    row_index: int
    prompt: str
    reference_answer: str | None = None
    source_fields: dict[str, str] = field(default_factory=dict)


@dataclass
class ModelTrial:
    row_index: int
    prompt: str
    reference_answer: str | None
    model_response: str | None
    latency_seconds: float | None = None
    error: str | None = None
    similarity_to_reference: float | None = None


@dataclass
class AdoptionDecision:
    action: str  # accept | review | reject
    final_score: float
    data_value_score: float
    model_gap_score: float
    training_types: list[TrainingType]
    reasons: list[str] = field(default_factory=list)
    notes: str = ""
    llm_decision: dict[str, Any] | None = None


@dataclass
class DatasetReportItem:
    dataset_id: str
    score: DatasetScore
    classification: DatasetClassification
    risk_assessment: RiskAssessment
    inspection: DatasetInspection
    web_overview: DatasetWebOverview | None = None
    model_trials: list[ModelTrial] = field(default_factory=list)
    adoption_decision: AdoptionDecision | None = None


def to_jsonable(obj: Any) -> Any:
    """Convert dataclasses/enums/sets/objects into JSON-serializable values."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
