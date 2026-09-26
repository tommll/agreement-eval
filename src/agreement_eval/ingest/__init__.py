"""Dataset loaders.

Adding a dataset means: a schema in schemas.py, a loader here that returns
Document objects, and an entry in LOADERS. Nothing downstream changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List

from ..documents import Document
from .funsd import load_funsd
from .sroie import load_sroie
from .synthetic import make_synthetic


def _sroie(**kw: Any) -> List[Document]:
    return load_sroie(
        split=kw.get("split") or "test",
        limit=kw.get("limit"),
        source=kw.get("source") or "api",
        download_images=bool(kw.get("download_images")),
        images_dir=kw.get("images_dir"),
        parquet_path=kw.get("parquet_path"),
    )


def _funsd(**kw: Any) -> List[Document]:
    root = kw.get("root")
    if not root:
        raise ValueError("funsd ingestion needs --root pointing at the extracted dataset")
    return load_funsd(Path(root), split=kw.get("split") or "testing", limit=kw.get("limit"))


def _synthetic(**kw: Any) -> List[Document]:
    return make_synthetic(n=kw.get("limit") or 60, seed=int(kw.get("seed") or 7))


LOADERS: Dict[str, Callable[..., List[Document]]] = {
    "sroie": _sroie,
    "funsd": _funsd,
    "synthetic": _synthetic,
}


def load_dataset(name: str, **kwargs: Any) -> List[Document]:
    try:
        loader = LOADERS[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; have {sorted(LOADERS)}") from None
    return loader(**kwargs)


__all__ = ["LOADERS", "load_dataset", "load_sroie", "load_funsd", "make_synthetic"]
