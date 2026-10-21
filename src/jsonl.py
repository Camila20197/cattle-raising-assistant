"""Read and write JSON Lines files (one JSON object per line).

Shared by ingestion, the API and the evaluation scripts, which previously
each had their own copy of the same loader.
"""

import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    """Load every non-empty line of a JSON Lines file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(records: list[dict], path: Path) -> None:
    """Save records as JSON Lines, creating the parent folder if needed.

    JSON Lines is convenient for data pipelines because each line is an
    independent record and can be processed without loading the whole file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
