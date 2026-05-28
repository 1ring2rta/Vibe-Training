from __future__ import annotations

from pathlib import Path


def normalize_workspace_path(root: str | Path, value: str | Path) -> Path:
    """Resolve action/tool paths under a workspace and strip duplicated prefixes."""
    root = Path(root).resolve()
    raw = Path(str(value)).expanduser()
    if raw.is_absolute():
        return raw

    text = str(raw).replace("\\", "/").lstrip("./")
    root_parts = list(root.parts)
    for n in range(min(len(root_parts), len(raw.parts)), 0, -1):
        suffix = root_parts[-n:]
        if list(raw.parts[:n]) == suffix:
            raw = Path(*raw.parts[n:]) if len(raw.parts) > n else Path(".")
            text = str(raw).replace("\\", "/")
            break

    marker = f"runs/{root.name}/"
    if text.startswith(marker):
        raw = Path(text[len(marker):])
    elif text == f"runs/{root.name}":
        raw = Path(".")
    return root / raw


def workspace_relative_path(root: str | Path, value: str | Path) -> str:
    path = normalize_workspace_path(root, value)
    try:
        return str(path.resolve().relative_to(Path(root).resolve()))
    except Exception:
        return str(value)
