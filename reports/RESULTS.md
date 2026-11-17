# Measured results

Results use the ULB/Worldline credit-card fraud benchmark through OpenML dataset 1597. The source is anonymized and the three-hour label delay is simulated. Model selection uses validation average precision; final-test results were not used to select the model.

| Model | Validation AP | Test AP | Test ROC-AUC | Precision at 10 reviews/day | Recall at 10 reviews/day |
|---|---:|---:|---:|---:|---:|
| logistic | 0.5844 | 0.6604 | 0.9729 | 100.0% | 13.3% |
| boosted | 0.7127 | 0.7736 | 0.9838 | 100.0% | 13.3% |

Selected model: **boosted**. The chronological run used 157,147 training rows, 31,813 validation rows and 56,962 final-test rows. The final test fraud prevalence was 0.13%. Capacity metrics are based on a 10-review daily queue and are sensitive to the benchmark's short two-day time range; they are not operational performance claims.

## Local API load measurements

The FastAPI service was run locally on Windows with Uvicorn and the selected ULB model. The replay utility sent 1,000 unique public benchmark transactions per run. These measurements include HTTP client and loopback overhead; they are a single workstation observation, not a production capacity claim.

| Replay scenario | Concurrency | Throughput (req/s) | p50 (ms) | p95 (ms) | Errors |
|---|---:|---:|---:|---:|---:|
| Normal | 1 | 31.13 | 31.69 | 49.67 | 0 / 1,000 |
| Normal | 16 | 284.22 | 54.30 | 59.62 | 0 / 1,000 |
| Amount ×4 and V1 +3 shift | 16 | 75.55 | 227.15 | 328.08 | 0 / 1,000 |

The shifted run is slower because monitoring work is triggered on the drifted request path; this is a signal to profile and decouple monitoring before claiming production readiness. The monitor observed 1,000 events and 29 feature drift measurements after replay. Model pointer activation and rollback were exercised locally between the real-data model and a clearly labeled synthetic demo model; the real-data model was restored afterward. No cloud deployment or cloud cost was measured, so no cost-per-request figure is claimed. SQLite, single-process serving, missing authentication, and missing rate limits keep this project at local-demo scope.
