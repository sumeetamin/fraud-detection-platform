# Measured results

Synthetic transactions; seed 42. Selection uses validation average precision. Test results were not used to change the selected model.

| Model | Validation AP | Test AP | Test ROC-AUC | Precision at 10 reviews/day | Recall at 10 reviews/day |
|---|---:|---:|---:|---:|---:|
| logistic | 0.0635 | 0.1129 | 0.7533 | 7.9% | 39.6% |
| boosted | 0.0514 | 0.1142 | 0.7962 | 7.5% | 37.5% |

Selected model: **logistic**. Final test fraud prevalence: 1.5%. The boosted model's slightly better final-test AP does not justify retroactively selecting it. Most reviewed transactions remain false positives, which is an operational limitation rather than a hidden metric.

A 120-request sequential local HTTP smoke run measured median latency 31.45 ms and p95 latency 46.93 ms. These values include local HTTP overhead; they are not a concurrent load or production SLA benchmark.
