"""Existence task."""

import math
import re
from typing import Any, ClassVar

import pandas as pd
from sklearn.metrics import roc_auc_score

from .base import Task


class Existence(Task[bool]):
    default_prompt = "Are there any {plural} in this image? Answer yes or no."
    default_answer_tokens: ClassVar[dict[str, list[str]]] = {
        "yes": ["yes"],
        "no": ["no"],
    }

    def parse_response(self, response: str) -> bool | None:
        match = re.search(r"\b(yes|no|[yn])\b", response, re.IGNORECASE)

        if match is None:
            return None
        return match.group().lower().startswith("y")

    def score(self, prediction: bool | None, expected: bool) -> dict[str, float]:
        if prediction is None:
            return {"valid": 0.0, "correct": float("nan")}

        correct = float(prediction == expected)
        return {"valid": 1.0, "correct": correct}

    def score_tokens(
        self, tokens: list[dict[str, Any]], expected: bool
    ) -> dict[str, float]:
        """P(yes) from the first token's alternatives, renormalized over yes + no."""
        top = tokens[0]["top"] if tokens else []

        mass = dict.fromkeys(self.answer_tokens, 0.0)
        for alternative in top:
            for key, variants in self.answer_tokens.items():
                if alternative["token"].strip().lower() in variants:
                    mass[key] += math.exp(alternative["logprob"])

        total = mass["yes"] + mass["no"]
        if total == 0:
            return {"covered": 0.0, "p_yes": float("nan")}
        return {"covered": 1.0, "p_yes": mass["yes"] / total}

    def summarize(self, rows: pd.DataFrame, metric_cols: list[str]) -> pd.Series:
        summary = super().summarize(rows, metric_cols)
        if "p_yes" not in rows.columns:
            return summary

        scored = rows.dropna(subset="p_yes")
        labels = scored["expected"].astype(bool)
        summary["auroc"] = (
            roc_auc_score(labels, scored["p_yes"])
            if labels.nunique() == 2
            else float("nan")
        )

        said_yes = scored["output"].map(self.parse_response).dropna().astype(bool)
        leans_yes = scored.loc[said_yes.index, "p_yes"] >= 0.5
        summary["disagreement"] = (
            (leans_yes != said_yes).mean() if len(said_yes) else float("nan")
        )
        return summary
