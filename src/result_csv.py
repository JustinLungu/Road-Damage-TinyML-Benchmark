import csv
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def save_result_csv(
    result: Any,
    output_path: Path,
    key_fields: tuple[str, ...],
) -> None:
    """Write one dataclass result, replacing a row with the same identity."""

    if not is_dataclass(result) or isinstance(result, type):
        raise TypeError("result must be a dataclass instance.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_row = asdict(result)
    previous_rows = []

    if output_path.is_file() and output_path.stat().st_size:
        with output_path.open(newline="", encoding="utf-8") as input_file:
            reader = csv.DictReader(input_file)
            if reader.fieldnames != list(result_row):
                raise ValueError(
                    f"Unexpected columns in {output_path}. Use --overwrite to "
                    "start a new result table."
                )
            previous_rows = [
                row
                for row in reader
                if any(row[field] != str(result_row[field]) for field in key_fields)
            ]

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(result_row))
        writer.writeheader()
        writer.writerows(previous_rows)
        writer.writerow(result_row)
