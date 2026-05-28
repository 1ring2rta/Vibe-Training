from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote

import requests
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


from autopilot.models import DatasetFileInfo, DatasetWebOverview

DATASET_VIEWER_BASE = "https://datasets-server.huggingface.co"
HF_BASE = "https://huggingface.co"
SPLIT_PRIORITY = ["train", "validation", "valid", "dev", "test"]


def _jsonable_row(value: Any, max_str_len: int = 4000) -> Any:
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


def _safe_get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _compact_markdown(text: str, max_chars: int = 7000) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    return text[:max_chars] + ("...[truncated]" if len(text) > max_chars else "")


def _extract_rows(payload: dict[str, Any], max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in payload.get("rows") or payload.get("first_rows") or []:
        if isinstance(item, dict) and isinstance(item.get("row"), dict):
            rows.append(_jsonable_row(item["row"]))
        elif isinstance(item, dict):
            rows.append(_jsonable_row(item))
        if len(rows) >= max_rows:
            break
    return rows


def _features_from_payload(payload: dict[str, Any]) -> dict[str, str]:
    features: dict[str, str] = {}
    for item in payload.get("features") or []:
        if isinstance(item, dict) and item.get("name") is not None:
            features[str(item["name"])] = json.dumps(item.get("type"), ensure_ascii=False)
    return features


def choose_config_split(entries: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    if not entries:
        return None, None

    def rank(entry: dict[str, Any]) -> tuple[int, int]:
        config = str(entry.get("config") or "")
        split = str(entry.get("split") or "")
        config_rank = 0 if config in {"default", "main", "all"} else 1
        try:
            split_rank = SPLIT_PRIORITY.index(split.lower())
        except ValueError:
            split_rank = len(SPLIT_PRIORITY)
        return config_rank, split_rank

    best = sorted(entries, key=rank)[0]
    return best.get("config"), best.get("split")


class HFDatasetBrowser:
    """Browse public HF dataset pages through Hub + Dataset Viewer APIs.

    This is read-only: it fetches card text, repository file names, dataset viewer
    split metadata, and a small examples preview.
    """

    def __init__(self, token: str | None = None, timeout: float = 20.0, endpoint: str | None = None, viewer_base: str | None = None) -> None:
        self.token = token
        self.timeout = timeout
        self.endpoint = endpoint.rstrip("/") if endpoint else None
        self.hf_base = self.endpoint or HF_BASE
        self.viewer_base = (viewer_base or DATASET_VIEWER_BASE).rstrip("/")
        try:
            self.api = HfApi(token=token, endpoint=self.endpoint) if self.endpoint else HfApi(token=token)
        except TypeError:  # older huggingface_hub versions may not accept endpoint
            self.api = HfApi(token=token)

    @property
    def headers(self) -> dict[str, str]:
        if not self.token:
            return {}
        return {"Authorization": f"Bearer {self.token}"}

    def _viewer_get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.viewer_base}/{endpoint.lstrip('/')}"
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=self.timeout)
            if response.status_code >= 400:
                return {"error": f"HTTP {response.status_code}: {response.text[:500]}"}
            try:
                data = response.json()
                return data if isinstance(data, dict) else {"data": data}
            except Exception as exc:
                return {"error": f"Invalid JSON: {exc}"}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _read_card(self, dataset_id: str) -> str:
        # Raw README URL is the closest equivalent to opening the HF dataset page
        # programmatically. Some repos have no README; we return empty in that case.
        url = f"{self.hf_base}/datasets/{quote(dataset_id, safe='/')}/raw/main/README.md"
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            if response.status_code < 400:
                return _compact_markdown(response.text)
        except Exception:
            pass
        return ""

    def _list_files(self, dataset_id: str, max_files: int) -> list[DatasetFileInfo]:
        try:
            info = self.api.dataset_info(dataset_id, files_metadata=True)
            siblings = _safe_get(info, "siblings", []) or []
        except Exception:
            siblings = []
        files: list[DatasetFileInfo] = []
        for sibling in siblings[:max_files]:
            path = _safe_get(sibling, "rfilename") or _safe_get(sibling, "path")
            if not path:
                continue
            lfs_obj = _safe_get(sibling, "lfs")
            size = _safe_get(sibling, "size")
            if size is None and isinstance(lfs_obj, dict):
                size = lfs_obj.get("size")
            files.append(
                DatasetFileInfo(
                    path=str(path),
                    size=int(size) if isinstance(size, int) else None,
                    blob_id=str(_safe_get(sibling, "blob_id")) if _safe_get(sibling, "blob_id") else None,
                    lfs=bool(lfs_obj) if lfs_obj is not None else None,
                )
            )
        return files

    def browse(self, dataset_id: str, sample_size: int = 20, max_files: int = 60) -> DatasetWebOverview:
        hub_url = f"{self.hf_base}/datasets/{dataset_id}"
        overview = DatasetWebOverview(dataset_id=dataset_id, hub_url=hub_url)
        errors: list[str] = []

        try:
            overview.card_excerpt = self._read_card(dataset_id)
        except Exception as exc:
            errors.append(f"card: {type(exc).__name__}: {exc}")

        try:
            overview.files = self._list_files(dataset_id, max_files=max_files)
        except Exception as exc:
            errors.append(f"files: {type(exc).__name__}: {exc}")

        status = self._viewer_get("is-valid", {"dataset": dataset_id})
        overview.viewer_status = status
        if status.get("error"):
            errors.append(f"viewer: {status['error']}")

        splits_payload = self._viewer_get("splits", {"dataset": dataset_id})
        raw_splits = splits_payload.get("splits") or []
        overview.configs_splits = [s for s in raw_splits if isinstance(s, dict)]
        config, split = choose_config_split(overview.configs_splits)
        overview.selected_config = config
        overview.selected_split = split

        if config and split:
            params = {"dataset": dataset_id, "config": config, "split": split}
            first_rows = self._viewer_get("first-rows", params)
            rows = _extract_rows(first_rows, max_rows=sample_size)
            if not rows:
                rows_payload = self._viewer_get("rows", {**params, "offset": 0, "length": min(100, sample_size)})
                rows = _extract_rows(rows_payload, max_rows=sample_size)
            overview.example_rows = rows
            if first_rows.get("error") and not rows:
                errors.append(f"examples: {first_rows['error']}")
        elif overview.configs_splits:
            errors.append("Could not choose config/split from viewer metadata.")
        else:
            if splits_payload.get("error"):
                errors.append(f"splits: {splits_payload['error']}")
            else:
                errors.append("No Dataset Viewer split metadata available.")

        overview.browse_error = "; ".join(errors) if errors else None
        return overview


def features_from_viewer_examples(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, str]]:
    if not rows:
        return [], {}
    columns = list(rows[0].keys())
    features = {c: type(rows[0].get(c)).__name__ for c in columns}
    return columns, features
