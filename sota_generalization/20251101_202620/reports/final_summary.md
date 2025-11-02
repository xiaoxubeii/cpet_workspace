# Generalization Sweep Summary

- Generated at 2025-11-01T12:27:22.459745+00:00

## Stage 1 – Fast Screening

| Rank | Model | Seed | MAE | R2 | RMSE |
|---|---|---|---|---|---|
| 1 | cpet_former_prototype_align | 17 | 8.3877 | -5.2176 | 9.1675 |
| 2 | cpet_former_mixstyle | 17 | 9.0953 | -6.1212 | 9.8110 |
| 3 | cpet_former_center_aware_mixstyle | 17 | 9.9712 | -7.3901 | 10.6493 |
| 4 | cpet_former_center_adaptive_adapter | 17 | 10.2114 | -7.7126 | 10.8520 |
| 5 | cpet_former_consistency_regularized | 17 | 10.2568 | -7.8169 | 10.9168 |
| 6 | cpet_former_center_invariant_layer_norm | 17 | 10.7305 | -8.5127 | 11.3394 |
| 7 | cpet_former_dann_regression | 17 | 11.1926 | -9.3535 | 11.8299 |
| 8 | cpet_former_stochastic_router | 17 | 12.5897 | -11.7355 | 13.1204 |

## Stage 2 – Stability (mean +/- std)

| Rank | Model | Seeds | MAE | R2 | RMSE |
|---|---|---|---|---|---|
| 1 | cpet_former_prototype_align | 2 | 10.9069 +/- 0.6580 | -8.8704 +/- 1.1019 | 11.5325 +/- 0.6457 |
| 2 | cpet_former_mixstyle | 2 | 11.1062 +/- 0.8463 | -9.1582 +/- 1.3905 | 11.6902 +/- 0.8039 |
| 3 | cpet_former_center_aware_mixstyle | 2 | 11.4379 +/- 0.6631 | -9.8025 +/- 1.1061 | 12.0678 +/- 0.6194 |
