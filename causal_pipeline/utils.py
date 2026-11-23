"""Filesystem helpers for pipeline artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd
from matplotlib.figure import Figure

FIGURE_DPI = 300
TEXT_ENCODING = "utf-8"
DATAFRAME_INDEX = False
FIGURE_FORMATS = frozenset({"png", "pdf"})
DEFAULT_FIGURE_FORMAT = "png"


def figure_file(stem: str, figure_format: str = DEFAULT_FIGURE_FORMAT) -> str:
    """File name for a matplotlib artifact. ``figure_format`` is ``png`` or ``pdf``."""
    token = figure_format.lower().removeprefix(".")
    if token not in FIGURE_FORMATS:
        allowed = ", ".join(sorted(FIGURE_FORMATS))
        raise ValueError(f"figure_format must be one of {allowed}, got {figure_format!r}.")
    if not stem or any(character in stem for character in "\\/"):
        raise ValueError(f"figure stem must be a single file name, got {stem!r}.")
    return f"{stem}.{token}"


class HtmlWritable(Protocol):
    def write_html(self, file: str) -> None:
        """Write an interactive figure to an HTML file."""


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_dataframe(path: str | Path, df: pd.DataFrame) -> Path:
    destination = Path(path)
    ensure_directory(destination.parent)
    if destination.suffix == ".csv":
        df.to_csv(destination, index=DATAFRAME_INDEX)
    elif destination.suffix == ".parquet":
        df.to_parquet(destination, index=DATAFRAME_INDEX)
    else:
        raise ValueError(f"Unsupported dataframe extension: {destination.suffix}")
    return destination


def read_dataframe(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix == ".csv":
        return pd.read_csv(source)
    if source.suffix == ".parquet":
        return pd.read_parquet(source)
    raise ValueError(f"Unsupported dataframe extension: {source.suffix}")


def write_text(path: str | Path, text: str) -> Path:
    destination = Path(path)
    ensure_directory(destination.parent)
    destination.write_text(text, encoding=TEXT_ENCODING)
    return destination


def save_figure(path: str | Path, figure: Figure) -> Path:
    destination = Path(path)
    ensure_directory(destination.parent)
    figure.savefig(
        destination,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    return destination


def write_html(path: str | Path, figure: HtmlWritable) -> Path:
    destination = Path(path)
    ensure_directory(destination.parent)
    figure.write_html(str(destination))
    return destination
