import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time

import numpy as np
import pandas as pd
import requests

from .core import ROOT, FEATURES


def main():
    parser = argparse.ArgumentParser(description="Replay public benchmark rows against a local fraud API.")
    parser.add_argument("--url", default="http://127.0.0.1:8003")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--incident", action="store_true")
    parser.add_argument("--api-key-env", help="Environment variable holding the API key, if the service requires one.")
    args = parser.parse_args()
    if args.limit < 1 or args.concurrency < 1:
        parser.error("--limit and --concurrency must be positive integers.")

    df = pd.read_csv(ROOT / "data/test_transactions.csv").head(args.limit)
    if args.incident:
        df["amount"] *= 4
        df["v1"] += 3
    if df.empty:
        parser.error("No replay rows found in data/test_transactions.csv.")
    headers = {}
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            parser.error(f"Environment variable {args.api_key_env} is empty or unset.")
        headers["X-API-Key"] = api_key

    def send(item):
        index, row = item
        data = {
            "transaction_id": f"{'SHIFT' if args.incident else 'REPLAY'}-{index}-{row.transaction_id}",
            "features": {key: float(row[key]) for key in FEATURES},
        }
        started = time.perf_counter()
        try:
            response = requests.post(f"{args.url.rstrip('/')}/predict", json=data, headers=headers, timeout=30)
            elapsed_ms = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            return elapsed_ms, None
        except requests.RequestException as exc:
            return (time.perf_counter() - started) * 1000, str(exc)

    started = time.perf_counter()
    outcomes = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(send, item) for item in df.iterrows()]
        for future in as_completed(futures):
            outcomes.append(future.result())
    elapsed_seconds = time.perf_counter() - started
    timings = [duration for duration, _ in outcomes]
    errors = [error for _, error in outcomes if error]
    print(json.dumps({
        "requests": len(outcomes),
        "successful_requests": len(outcomes) - len(errors),
        "errors": len(errors),
        "concurrency": args.concurrency,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "throughput_requests_per_second": round((len(outcomes) - len(errors)) / elapsed_seconds, 2),
        "p50_ms": round(float(np.percentile(timings, 50)), 2),
        "p95_ms": round(float(np.percentile(timings, 95)), 2),
        "error_samples": errors[:3],
        "incident_shift": args.incident,
        "note": "Local benchmark includes client/network overhead. Results depend on host hardware and do not establish production capacity.",
    }, indent=2))


if __name__ == "__main__":
    main()
