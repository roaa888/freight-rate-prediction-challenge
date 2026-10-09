# Reproducing the Freight Rate Submission

Run these commands from the repository root with Python 3.10+:

```bash
python -m pip install -r requirements.txt
python -m src.predict
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python -m src.render_report
```

`src.predict` fits the frozen HistGradientBoosting pipeline on all 48,000 labeled rows in `data/train_test.csv`. It writes `validation_predictions.csv` in the template's `load_id` order and fills the existing `predicted_rate` column of `data/december_chart_inputs.csv`. The command checks the output schemas, row counts, IDs, and positive finite rates before writing. It can be rerun after the December file has been filled.

The unchanged scorer checks both CSVs and creates `scorer_results/candidate_december.png`. It does not calculate local validation accuracy; the hidden metric is calculated after submission. `src.render_report` builds `report/report_draft.pdf` from `report/report_draft.md` and the scorer chart. The untouched blank December assessment input is preserved as `data/december_chart_inputs_original.csv` for reference; the scorer uses the filled `data/december_chart_inputs.csv`.

To reproduce the earlier experiments, run `python -m src.train`, `python -m src.diagnostics`, and `python -m src.catboost_experiment`. The final model uses only fields available for December inference: pickup, delivery, distance, equipment, weight, and date. Learned lane counts and preprocessing are fitted on training rows only.
