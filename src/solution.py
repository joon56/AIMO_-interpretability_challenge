"""Codabench entry point. Each call: one model config, (normally) one problem, fresh process."""

import sys
import traceback

from aimo_robust.predict import predict_one


def are_robust(model_id: str, problems: list[str], reasoning_effort: str = "default") -> list[bool]:
    efforts = reasoning_effort if isinstance(reasoning_effort, list) else [reasoning_effort] * len(problems)
    preds = []
    for problem, effort in zip(problems, efforts):
        try:
            preds.append(bool(predict_one(model_id, problem, effort or "default")))
        except Exception:
            # An invalid prediction scores as wrong; a constant guess is never worse.
            traceback.print_exc(file=sys.stderr)
            preds.append(True)
    return preds
