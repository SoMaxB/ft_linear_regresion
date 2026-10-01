"""Generate a dependency-free SVG plot of the regression dataset."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from collections.abc import Sequence
import sys

# Make direct execution (`python3 bonus/plot.py`) behave like module execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import Dataset, DatasetError, load_dataset


DEFAULT_OUTPUT = "bonus/output/dataset.svg"
WIDTH = 900
HEIGHT = 600
MARGIN_LEFT = 90
MARGIN_RIGHT = 30
MARGIN_TOP = 55
MARGIN_BOTTOM = 75
TICK_COUNT = 5


def create_dataset_svg(
    dataset: Dataset,
    *,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> str:
    """Return an SVG scatter plot for a validated dataset."""
    if width <= MARGIN_LEFT + MARGIN_RIGHT:
        raise ValueError("SVG width is too small for the plot margins")
    if height <= MARGIN_TOP + MARGIN_BOTTOM:
        raise ValueError("SVG height is too small for the plot margins")

    x_min, x_max = _expanded_range(dataset.mileage)
    y_min, y_max = _expanded_range(dataset.prices)
    plot_width = width - MARGIN_LEFT - MARGIN_RIGHT
    plot_height = height - MARGIN_TOP - MARGIN_BOTTOM
    x_axis = height - MARGIN_BOTTOM
    y_axis = MARGIN_LEFT

    def x_position(value: float) -> float:
        return MARGIN_LEFT + (value - x_min) / (x_max - x_min) * plot_width

    def y_position(value: float) -> float:
        return MARGIN_TOP + (y_max - value) / (y_max - y_min) * plot_height

    elements: list[str] = []
    elements.append(f'<rect width="{width}" height="{height}" fill="white"/>')

    for value in _ticks(x_min, x_max):
        x = x_position(value)
        elements.append(
            f'<line x1="{x:.2f}" y1="{MARGIN_TOP}" x2="{x:.2f}" '
            f'y2="{x_axis}" stroke="#e5e7eb"/>'
        )
        elements.append(
            f'<text x="{x:.2f}" y="{x_axis + 25}" text-anchor="middle" '
            f'font-size="12" fill="#374151">{_format_number(value)}</text>'
        )

    for value in _ticks(y_min, y_max):
        y = y_position(value)
        elements.append(
            f'<line x1="{y_axis}" y1="{y:.2f}" x2="{width - MARGIN_RIGHT}" '
            f'y2="{y:.2f}" stroke="#e5e7eb"/>'
        )
        elements.append(
            f'<text x="{y_axis - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="12" fill="#374151">{_format_number(value)}</text>'
        )

    elements.extend(
        (
            f'<line x1="{y_axis}" y1="{MARGIN_TOP}" x2="{y_axis}" '
            f'y2="{x_axis}" stroke="#111827" stroke-width="2"/>',
            f'<line x1="{y_axis}" y1="{x_axis}" x2="{width - MARGIN_RIGHT}" '
            f'y2="{x_axis}" stroke="#111827" stroke-width="2"/>',
        )
    )

    for mileage, price in zip(dataset.mileage, dataset.prices):
        elements.append(
            f'<circle cx="{x_position(mileage):.2f}" cy="{y_position(price):.2f}" '
            'r="5" fill="#2563eb"/>'
        )

    elements.extend(
        (
            f'<text x="{width / 2:.2f}" y="28" text-anchor="middle" '
            'font-size="20" font-weight="bold" fill="#111827">'
            'Mileage and price dataset</text>',
            f'<text x="{width / 2:.2f}" y="{height - 18}" text-anchor="middle" '
            'font-size="14" fill="#111827">Mileage (km)</text>',
            f'<text x="18" y="{height / 2:.2f}" text-anchor="middle" '
            'font-size="14" fill="#111827" '
            f'transform="rotate(-90 18 {height / 2:.2f})">Price</text>',
        )
    )

    body = "\n  ".join(elements)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n  {body}\n</svg>\n'
    )


def write_dataset_svg(dataset: Dataset, output_path: str | Path) -> None:
    """Write a dataset SVG, creating its parent directory when needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(create_dataset_svg(dataset), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Load a CSV dataset and write its SVG scatter plot."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)

    try:
        dataset = load_dataset(arguments.data)
        write_dataset_svg(dataset, arguments.output)
    except (DatasetError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Dataset plot written to {arguments.output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an SVG plot of the mileage-to-price dataset."
    )
    parser.add_argument(
        "--data",
        default="data.csv",
        help="CSV dataset path (default: data.csv)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"SVG output path (default: {DEFAULT_OUTPUT})",
    )
    return parser


def _expanded_range(values: Sequence[float]) -> tuple[float, float]:
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum
    padding = span * 0.05 if span else max(abs(minimum) * 0.05, 1.0)
    return minimum - padding, maximum + padding


def _ticks(minimum: float, maximum: float) -> tuple[float, ...]:
    step = (maximum - minimum) / (TICK_COUNT - 1)
    return tuple(minimum + index * step for index in range(TICK_COUNT))


def _format_number(value: float) -> str:
    if math.isclose(value, round(value)):
        return f"{value:,.0f}"
    return f"{value:,.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
