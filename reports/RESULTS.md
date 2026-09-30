# Measured results

Results use the ULB/Worldline credit-card fraud benchmark through OpenML dataset 1597. The source is anonymized and the three-hour label delay is simulated. Model selection uses validation average precision; final-test results were not used to select the model.

| Model | Validation AP | Test AP | Test ROC-AUC | Precision at 10 reviews/day | Recall at 10 reviews/day |
|---|---:|---:|---:|---:|---:|
| logistic | 0.5844 | 0.6604 | 0.9729 | 100.0% | 13.3% |
| boosted | 0.7127 | 0.7736 | 0.9838 | 100.0% | 13.3% |

Selected model: **boosted**. The chronological run used 157,147 training rows, 31,813 validation rows and 56,962 final-test rows. The final test fraud prevalence was 0.13%. Capacity metrics are based on a 10-review daily queue and are sensitive to the benchmark's short two-day time range; they are not operational performance claims.
