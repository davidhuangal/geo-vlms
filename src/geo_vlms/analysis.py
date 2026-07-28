import json
from pathlib import Path

import pandas as pd


def load_results(results_path) -> pd.DataFrame:
    """Read a raw `.jsonl` of records back into a DataFrame."""
    lines = Path(results_path).read_text().splitlines()
    results = [json.loads(x) for x in lines]

    results_df = pd.DataFrame(results)

    return results_df
