import json
import re
from pathlib import Path

import pandas as pd


def load_results(results_path):
    lines = Path(results_path).read_text().splitlines()
    results = [json.loads(x) for x in lines]

    results_df = pd.DataFrame(results)

    return results_df


def pivot_results(results_df):
    pivot = results_df.pivot(index="example_id", columns="condition", values="output")
    return pivot


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())
