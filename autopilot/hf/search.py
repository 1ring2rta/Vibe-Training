from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterable

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


from autopilot.models import DatasetSearchResult


OPEN_LICENSE_ALIASES = {
    "apache-2.0",
    "mit",
    "bsd-3-clause",
    "bsd-2-clause",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc0-1.0",
    "odc-by",
}


def _safe_get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def extract_license(dataset_info: Any) -> str | None:
    """Best-effort license extraction from DatasetInfo.card_data and tags."""
    card_data = _safe_get(dataset_info, "card_data")
    license_value = None
    if card_data is not None:
        if isinstance(card_data, dict):
            license_value = card_data.get("license") or card_data.get("License")
        else:
            license_value = getattr(card_data, "license", None)
    if isinstance(license_value, list):
        license_value = license_value[0] if license_value else None
    if license_value:
        return str(license_value).lower()

    tags = _safe_get(dataset_info, "tags", []) or []
    for tag in tags:
        tag_s = str(tag).lower()
        if tag_s.startswith("license:"):
            return tag_s.split(":", 1)[1]
        if tag_s in OPEN_LICENSE_ALIASES or "license" in tag_s:
            return tag_s.replace("license:", "")
    return None


def info_to_search_result(dataset_info: Any) -> DatasetSearchResult:
    dataset_id = _safe_get(dataset_info, "id") or _safe_get(dataset_info, "datasetId")
    if not dataset_id:
        raise ValueError(f"Could not read dataset id from DatasetInfo: {dataset_info!r}")
    last_modified = _safe_get(dataset_info, "last_modified")
    if last_modified is not None:
        last_modified = str(last_modified)
    return DatasetSearchResult(
        dataset_id=str(dataset_id),
        author=_safe_get(dataset_info, "author"),
        downloads=_safe_get(dataset_info, "downloads"),
        likes=_safe_get(dataset_info, "likes"),
        tags=[str(t) for t in (_safe_get(dataset_info, "tags", []) or [])],
        license=extract_license(dataset_info),
        last_modified=last_modified,
        gated=_safe_get(dataset_info, "gated"),
        private=_safe_get(dataset_info, "private"),
    )


class HuggingFaceDatasetSearcher:
    """Search Hugging Face datasets and return lightweight metadata."""

    def __init__(self, token: str | None = None, endpoint: str | None = None) -> None:
        self.api = HfApi(token=token, endpoint=endpoint) if endpoint else HfApi(token=token)
        self.endpoint = endpoint

    def search(
        self,
        query: str,
        limit: int = 20,
        sort: str = "downloads",
    ) -> list[DatasetSearchResult]:
        kwargs = {"search": query, "sort": sort, "limit": limit, "full": True}
        try:
            raw = self.api.list_datasets(**kwargs, direction=-1)
        except TypeError:
            raw = self.api.list_datasets(**kwargs)
        except Exception:
            # Some versions do not accept `full` for list_datasets.
            kwargs.pop("full", None)
            try:
                raw = self.api.list_datasets(**kwargs, direction=-1)
            except TypeError:
                raw = self.api.list_datasets(**kwargs)

        results: list[DatasetSearchResult] = []
        for item in raw:
            try:
                results.append(info_to_search_result(item))
            except Exception:
                continue
        return results

    def search_many(
        self,
        queries: Iterable[str],
        per_query_limit: int = 20,
        max_total: int = 50,
        sort: str = "downloads",
    ) -> list[DatasetSearchResult]:
        dedup: OrderedDict[str, DatasetSearchResult] = OrderedDict()
        self.last_errors: list[str] = []
        for query in queries:
            if len(dedup) >= max_total:
                break
            try:
                results = self.search(query=query, limit=per_query_limit, sort=sort)
            except Exception as exc:
                self.last_errors.append(str(exc))
                continue
            for result in results:
                if result.dataset_id not in dedup:
                    dedup[result.dataset_id] = result
                if len(dedup) >= max_total:
                    break
        return list(dedup.values())
