"""CSV loading and validation for the regression dataset."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


class DatasetError(ValueError):
    """Raised when a dataset cannot be used for training."""


@dataclass(frozen=True)
class Dataset:
    """Validated mileage and price observations."""

    mileage: tuple[float, ...]
    prices: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.mileage:
            raise DatasetError("Dataset must contain at least one observation")
        if len(self.mileage) != len(self.prices):
            raise DatasetError("Mileage and price columns must have the same length")

    def __len__(self) -> int:
        return len(self.mileage)


def load_dataset(path: str | Path = "data.csv") -> Dataset:
    """Load and validate a `km,price` CSV dataset."""
    dataset_path = Path(path)

    try:
        with dataset_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file)
            header = next(reader, None)
            if header != ["km", "price"]:
                raise DatasetError(
                    "Invalid header: expected exactly 'km,price'"
                )

            mileage: list[float] = []
            prices: list[float] = []
            for row in reader:
                line_number = reader.line_num
                if not row or all(not value.strip() for value in row):
                    continue
                if len(row) != 2:
                    raise DatasetError(
                        f"Line {line_number}: expected 2 columns, got {len(row)}"
                    )

                mileage.append(_parse_finite_number(row[0], line_number, "km"))
                prices.append(_parse_finite_number(row[1], line_number, "price"))
    except FileNotFoundError as error:
        raise DatasetError(f"Dataset not found: {dataset_path}") from error
    except OSError as error:
        raise DatasetError(f"Cannot read dataset '{dataset_path}': {error}") from error
    except csv.Error as error:
        raise DatasetError(f"Invalid CSV format in '{dataset_path}': {error}") from error

    return Dataset(tuple(mileage), tuple(prices))


def _parse_finite_number(value: str, line_number: int, column: str) -> float:
    try:
        number = float(value.strip())
    except ValueError as error:
        raise DatasetError(
            f"Line {line_number}, column '{column}': expected a number"
        ) from error

    if not math.isfinite(number):
        raise DatasetError(
            f"Line {line_number}, column '{column}': value must be finite"
        )
    return number
