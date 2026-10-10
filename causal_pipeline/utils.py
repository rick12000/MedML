"""Filesystem helpers for pipeline artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd
from matplotlib.figure import Figure

FIGURE_DPI = 300
TEXT_ENCODING = "utf-8"
DATAFRAME_INDEX = False


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
