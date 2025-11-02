# Generalization Sweep Summary

- Generated at 2025-11-01T13:27:13.496688+00:00

## Stage 1 – Fast Screening

| Rank | Model | Seed | MAE | R2 | RMSE |
|---|---|---|---|---|---|
| 1 | cpet_former_center_invariant_layer_norm | 17 | 0.8336 | 0.8664 | 1.2733 |
| 2 | cpet_former_stochastic_router | 17 | 0.8513 | 0.8417 | 1.3859 |
| 3 | cpet_former_consistency_regularized | 17 | 0.8591 | 0.8371 | 1.4056 |
| 4 | cpet_former_center_adaptive_adapter | 17 | 0.9301 | 0.8302 | 1.4350 |
| 5 | cpet_former_prototype_align | 17 | 0.9336 | 0.8322 | 1.4268 |
| 6 | cpet_former_center_aware_mixstyle | 17 | 0.9483 | 0.8264 | 1.4510 |
| 7 | cpet_former_mixstyle | 17 | 0.9563 | 0.8204 | 1.4759 |
| 8 | cpet_former_dann_regression | 17 | 1.0361 | 0.7969 | 1.5696 |

## Stage 2 – Stability (mean +/- std)

| Rank | Model | Seeds | MAE | R2 | RMSE |
|---|---|---|---|---|---|
| 1 | cpet_former_center_invariant_layer_norm | 3 | 0.8842 +/- 0.0108 | 0.8460 +/- 0.0101 | 1.4082 +/- 0.0457 |
| 2 | cpet_former_consistency_regularized | 3 | 0.8945 +/- 0.0153 | 0.8500 +/- 0.0052 | 1.3905 +/- 0.0240 |
| 3 | cpet_former_stochastic_router | 3 | 0.9019 +/- 0.0311 | 0.8529 +/- 0.0032 | 1.3768 +/- 0.0148 |
