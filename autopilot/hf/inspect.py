from __future__ import annotations

import itertools
import json
import os
from typing import Any

try:
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
except Exception:  # pragma: no cover - exercised when optional dependency is absent
    get_dataset_config_names = None  # type: ignore[assignment]
    get_dataset_split_names = None  # type: ignore[assignment]
    load_dataset = None  # type: ignore[assignment]

try:
    from huggingface_hub import HfApi
except Exception:  # pragma: no cover - optional dependency absent in offline tests
    class HfApi:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

        def list_datasets(self, *args, **kwargs):
            raise ImportError("huggingface_hub is not installed")

        def dataset_info(self, *args, **kwargs):
            raise ImportError("huggingface_hub is not installed")


from autopilot.hf.search import info_to_search_result
from autopilot.models import DatasetInspection, DatasetSearchResult


SPLIT_PRIORITY = ["train", "validation", "valid", "dev", "test"]


def _jsonable_row(value: Any, max_str_len: int = 4000) -> Any:
    """Make dataset rows safe to serialize and report."""
    if isinstance(value, dict):
        return {str(k): _jsonable_row(v, max_str_len=max_str_len) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_row(v, max_str_len=max_str_len) for v in value[:50]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > max_str_len:
            return value[:max_str_len] + "...[truncated]"
        return value
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)[:max_str_len]


def _features_to_dict(features: Any) -> dict[str, str]:
    if not features:
        return {}
    out: dict[str, str] = {}
    for key, value in getattr(features, "items", lambda: [])():
        out[str(key)] = str(value)
    return out


def _choose_split(split_names: list[str]) -> str | None:
    if not split_names:
        return None
    lower_to_original = {s.lower(): s for s in split_names}
    for preferred in SPLIT_PRIORITY:
        if preferred in lower_to_original:
            return lower_to_original[preferred]
    return split_names[0]


class DatasetInspector:
    """Load a tiny streaming sample from a Hugging Face dataset.

    The inspector intentionally uses streaming by default and does not trust remote
    dataset code unless explicitly requested by the caller.
    """

    def __init__(self, token: str | None = None, trust_remote_code: bool = False, endpoint: str | None = None) -> None:
        self.token = token
        self.trust_remote_code = trust_remote_code
        self.endpoint = endpoint.rstrip("/") if endpoint else None
        if self.endpoint:
            # `datasets`/`huggingface_hub` honor HF_ENDPOINT in many code paths;
            # set it here so streaming loads can use the same mirror as HfApi.
            os.environ.setdefault("HF_ENDPOINT", self.endpoint)
        try:
            self.api = HfApi(token=token, endpoint=self.endpoint) if self.endpoint else HfApi(token=token)
        except TypeError:  # older huggingface_hub versions may not accept endpoint
            self.api = HfApi(token=token)

    def get_metadata(self, dataset_id: str, fallback: DatasetSearchResult | None = None) -> DatasetSearchResult:
        try:
            info = self.api.dataset_info(dataset_id, files_metadata=False)
            return info_to_search_result(info)
        except Exception:
            if fallback is not None:
                return fallback
            return DatasetSearchResult(dataset_id=dataset_id)

    def list_configs(self, dataset_id: str, max_configs: int = 3) -> list[str | None]:
        if get_dataset_config_names is None:
            return [None]
        try:
            configs = get_dataset_config_names(
                dataset_id,
                token=self.token,
                trust_remote_code=self.trust_remote_code,
            )
            if not configs:
                return [None]
            # Prefer the default config, then keep a small bounded list.
            ordered: list[str] = []
            if "default" in configs:
                ordered.append("default")
            ordered.extend([c for c in configs if c not in ordered])
            return ordered[:max_configs]
        except Exception:
            return [None]

    def list_splits(self, dataset_id: str, config_name: str | None) -> list[str]:
        if get_dataset_split_names is None:
            return ["train"]
        try:
            kwargs: dict[str, Any] = {
                "path": dataset_id,
                "token": self.token,
                "trust_remote_code": self.trust_remote_code,
            }
            if config_name is not None:
                kwargs["name"] = config_name
            splits = get_dataset_split_names(**kwargs)
            return [str(s) for s in splits]
        except Exception:
            return ["train"]

    def inspect(
        self,
        dataset_id: str,
        sample_size: int = 20,
        search_metadata: DatasetSearchResult | None = None,
        max_configs: int = 3,
    ) -> DatasetInspection:
        metadata = self.get_metadata(dataset_id, fallback=search_metadata)
        if load_dataset is None:
            return DatasetInspection(
                dataset_id=dataset_id,
                config_name=None,
                split=None,
                columns=[],
                features={},
                sample_rows=[],
                metadata=metadata,
                load_error="The optional 'datasets' package is not installed; falling back to Dataset Viewer examples if available.",
            )

        last_error: str | None = None

        for config_name in self.list_configs(dataset_id, max_configs=max_configs):
            split = _choose_split(self.list_splits(dataset_id, config_name))
            if split is None:
                continue
            try:
                kwargs: dict[str, Any] = {
                    "path": dataset_id,
                    "split": split,
                    "streaming": True,
                    "token": self.token,
                    "trust_remote_code": self.trust_remote_code,
                }
                if config_name is not None:
                    kwargs["name"] = config_name
                ds = load_dataset(**kwargs)

                rows = [_jsonable_row(row) for row in itertools.islice(iter(ds), sample_size)]
                features = _features_to_dict(getattr(ds, "features", None))
                columns: list[str] = list(features.keys()) if features else []
                if not columns and rows:
                    columns = list(rows[0].keys())
                return DatasetInspection(
                    dataset_id=dataset_id,
                    config_name=config_name,
                    split=split,
                    columns=columns,
                    features=features,
                    sample_rows=rows,
                    metadata=metadata,
                    load_error=None,
                )
            except Exception as exc:
                last_error = f"config={config_name!r}: {type(exc).__name__}: {exc}"
                continue

        return DatasetInspection(
            dataset_id=dataset_id,
            config_name=None,
            split=None,
            columns=[],
            features={},
            sample_rows=[],
            metadata=metadata,
            load_error=last_error or "Could not load dataset sample.",
        )
