# Generalization Sweep Summary

- Generated at 2025-11-01T12:22:54.979343+00:00

## Stage 1 – Fast Screening

| Rank | Model | Seed | MAE | R2 | RMSE |
|---|---|---|---|---|---|
| 1 | cpet_former_center_adaptive_adapter | 17 | 7.4724 | -4.1128 | 8.3132 |
| 2 | cpet_former_mixstyle | 17 | 7.9965 | -4.7409 | 8.8090 |
| 3 | cpet_former_center_aware_mixstyle | 17 | 9.1248 | -6.2467 | 9.8971 |
| 4 | cpet_former_prototype_align | 17 | 9.6245 | -6.8267 | 10.2855 |
| 5 | cpet_former_center_invariant_layer_norm | 17 | 9.9631 | -7.3722 | 10.6379 |
| 6 | cpet_former_consistency_regularized | 17 | 10.4435 | -8.0731 | 11.0742 |
| 7 | cpet_former_stochastic_router | 17 | 12.4840 | -11.5366 | 13.0175 |

## Stage 2 – Stability (mean +/- std)

| Rank | Model | Seeds | MAE | R2 | RMSE |
|---|---|---|---|---|---|
| 1 | cpet_former_center_aware_mixstyle | 1 | 10.5643 +/- 0.0000 | -8.2780 +/- 0.0000 | 11.1986 +/- 0.0000 |
| 2 | cpet_former_center_adaptive_adapter | 1 | 11.1750 +/- 0.0000 | -9.2985 +/- 0.0000 | 11.7984 +/- 0.0000 |
| 3 | cpet_former_mixstyle | 1 | 11.4572 +/- 0.0000 | -9.7214 +/- 0.0000 | 12.0382 +/- 0.0000 |
