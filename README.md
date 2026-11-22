# ZIP-level housing affordability model

Predicting the share of households in a US ZIP code that are not housing-cost-burdened, for ZIPs the model has seen before and for ZIPs it has never seen. The two cases behave so differently that most of the work was recognising which one mattered.

## Run the baseline

| Evaluation | Baseline | RMSE |
| --- | --- | ---: |
| Seen ZIPs, May to June | Carry the previous month's value forward and adjust for average drift | 1.208 |
| Unseen ZIPs, five folds grouped by ZIP | LightGBM with ZIP and market features | 12.329 |

The baseline tests two different prediction settings. The competition score weights unseen ZIP error four times as heavily, so the later experiments focus there. A mean-only prediction on unseen ZIPs scores 28.467 RMSE.

```bash
python -m pip install -r requirements.txt
# Place the competition's train.csv, test.csv and sample_submission.csv under data/.
python src/baseline.py
```

The script prints both validation results and writes `outputs/v1_baseline.csv`. The competition files must be obtained separately.

## The problem

Regression on 6,830 ZIPs observed in May and June 2026 (13,660 rows), predicting July. The target is the percentage of a ZIP's households whose monthly housing cost is at or below 30% of the ZIP's median monthly income. Features cover rent, income, population and age, owners versus renters, household size and composition, and length of residence. Of the July ZIPs, 6,830 appear in training and 1,704 are new.

Error is RMSE, weighted `0.2 * RMSE(seen) + 0.8 * RMSE(new)`. Only the provided dataset is used; no outside data.

## What I found

**Seen ZIPs are nearly solved; new ZIPs are the whole problem.** The target barely moves month to month (May to June change has SD 1.2, r = 0.999). June's value plus the mean drift gives RMSE 1.21 on seen ZIPs. Predicting the mean for new ZIPs gives 28.5. So almost all of the weighted error, and all of the modelling effort, sits on new ZIPs.

**Validation has to split by ZIP.** A random row split puts the same ZIP in train and validation and reports an error that does not exist for new ZIPs. Every number below is 5-fold `GroupKFold` by ZIP.

**Within-group ranks beat raw features.** Replacing each feature with its percentile within the ZIP's city, metro, ZIP3 and state moved a single LightGBM from 11.74 to 11.23 new-ZIP RMSE, the largest single gain. Income alone is negatively related to the target (expensive coastal metros), so a ZIP's position within its own market carries more than its absolute level.

**A distribution shift between months, found by looking at the test features.** In July, adults, household size, generations, young-adult share and senior share step up by 0.16 to 0.37 SD against June, while May to June moved less than 0.06 SD. Models trained on May and June misread the July rows, and the rank features compared July ZIPs against June ones. `features.load(align=True)` subtracts the mean June-to-July change measured on seen ZIPs from every test row. Refitting the same models with it improved the independent held-out score by about 0.11, which was the largest gain in the second half of the project.

**Calibration was needed per component.** On seen ZIPs, where July is known to about 1.2, every model predicts above the July level because it learned June levels: +0.66 for the GBDT base and +1.7 to +2.1 for the neural nets. Without the alignment fix the neural net's bias was +3.4.

**Ablation, new-ZIP RMSE:**

| change | RMSE |
|---|---|
| raw features + group target encodings | 12.39 |
| + feature context (metro, state, ZIP3 medians over train and test, and each ZIP's deviation from them) | 11.81 |
| regularised LightGBM | 11.74 |
| CatBoost, depth 6 | 11.73 |
| NNLS blend of LightGBM and CatBoost, raw and logit target | 11.63 |
| + within-group percentile ranks, ZIP-vs-city ratios, kNN target means in rank space, quantile matching | 11.01 |
| + July alignment, MLPs on quantile-normalised features, tuned LightGBM, rank model (blend on saved out-of-fold predictions) | 10.71 |

Gains stopped at about 10.71: extra seeds, SVR, and more MLP variants moved the blend by under 0.01.

The 10.71 is an exploratory blend estimate. `src/combine.py` refits blend weights per fold using cached out-of-fold predictions. Upstream models were fitted once per original split. Predictions used to fit the blend in one split can come from models trained on ZIPs in that blend's evaluation fold. A fully nested estimate would regenerate the component predictions within each outer training fold.

**Things that did not help.** Huber loss, prediction stretching and isotonic calibration, city-level quantile matching, kNN in metro-rank space, an LGB stacker over out-of-fold predictions (11.20 against 11.04 for NNLS), and adding metro-level drift for seen ZIPs. New-ZIP errors are not spatially clustered (leave-one-out residual correlation within city, ZIP3 and metro is about 0), so residual-correction stages by geography had nothing to find.

**Where the remaining error is.** It is largest for small-sample ZIPs (14.7 RMSE against 9.9) and at the extremes, where the model pulls toward the middle. Inside a metro, rent has about 0 correlation with the target, while income, owner share and number of children matter. Owner costs, meaning home prices, are the likely hidden driver and are not in the data.

**Seen-ZIP drift.** The July level is predicted as June plus a per-state drift, shrunk toward the national mean, scaled by a level factor (ZIPs near 0% and above 90% move less) and then multiplied by about 2 to 3, because the downward trend continues into July. The multiplier was chosen against an external held-out check; its effect is small and the larger values were not clearly distinguishable from each other.

## Repository

```
src/features.py     feature building: group context, within-group ranks, July alignment
src/baseline.py     v1: June + drift for seen ZIPs, LightGBM with target encodings for new ZIPs
src/experiments.py  ablations, GroupKFold by ZIP
src/models.py       model definitions; mats.py, full_mats.py save fold and full-refit design matrices
src/push.py         the rank, kNN and quantile-matching feature stages
src/nn.py           MLPs on quantile-normalised features; svr.py, rankmodel.py, tune.py, tuned.py
src/combine.py      NNLS blend of cached fold predictions and final prediction file
src/seen.py         seen-ZIP drift variants
src/train_v*.py     earlier staged builds
```

## Later experiments

`python src/models.py` and `python src/train_v2.py` run intermediate modeling stages. The full blend needs cached fold arrays built by the research scripts, starting with `src/push.py` and `src/mats.py`. Some stages take about 10 minutes of mostly CatBoost on a laptop CPU. The baseline command above runs without those caches.

## Limits

The code was run as a research pipeline, not packaged: scripts are numbered by stage and share an `outputs/` directory. There are no unit tests. Seen-ZIP drift settings were tuned against an external held-out score on a subset of rows, so treat those multipliers as approximate.

## License

MIT. See `LICENSE`.
