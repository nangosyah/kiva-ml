# kiva-ml

An end-to-end Python ML pipeline, built as a tutorial example following the
structure of Posit's ["Building data pipelines in Python & R"](https://posit.co/blog/building-data-pipelines-in-python-r)
blog post, adapted to a single-language (Python) stack using `pins`,
`vetiver`, and Shiny for Python.

**Task:** classify a Kiva Uganda microloan into one of three sectors
(`Agriculture`, `Business`, `Personal`) from the borrower's free-text
description of what the loan will be used for.

## Pipeline stages

```
kiva_extract.py  -->  transform.py  -->  train.py  -->  evaluate.py
                                              |
                                              v
                                         deploy.py  -->  app.py
```

| Stage | Script | What it does |
|---|---|---|
| Extract | `kiva_extract.py` | Pulls Uganda loan listings from the public Kiva API (`api.kivaws.org`) and writes the raw data to `kiva_uganda.csv`. |
| Transform | `transform.py` | Tidies the raw data: engineers `description_length`, merges rare sectors (<20 obs) into `Other`, then groups sectors into 3 balanced classes. Writes the result to the `kiva_tidy_loans` pin on a local pins board (`pins_board/`). |
| Train | `train.py` | Splits into train/test (80/20, stratified), cross-validates a TF-IDF + `LogisticRegression` pipeline, fits it on the training split, and saves it as a `vetiver` model pin (`kiva_sector_model`). Also pins the untouched held-out test split (`kiva_sector_test`). |
| Evaluate | `evaluate.py` | One-time final evaluation of the pinned model against the held-out test pin. Run this only after model development is complete. |
| Deploy | `deploy.py` | Loads the model pin and serves it as a local FastAPI app via `vetiver.VetiverAPI`, exposing a `/predict` endpoint. |
| App | `app.py` | A Shiny for Python app with a text box for the loan description; calls the local `/predict` endpoint and displays the predicted sector. |

Model documentation lives in [`model_card.qmd`](model_card.qmd) (a Quarto
document following the [model cards](https://doi.org/10.1145/3287560.3287596)
format), covering intended use, metrics, and known limitations.

## Running it

Install dependencies (into whichever Python environment you're using):

```bash
pip install pandas requests scikit-learn pins vetiver shiny
```

Run the pipeline stages in order:

```bash
python kiva_extract.py   # writes kiva_uganda.csv
python transform.py      # writes kiva_tidy_loans pin
python train.py           # writes kiva_sector_model + kiva_sector_test pins
python evaluate.py        # one-time test-set evaluation, prints metrics
```

Then, in two separate terminals, start the API and the app:

```bash
# terminal 1
uvicorn deploy:app --port 8080

# terminal 2
shiny run app.py
```

Open the Shiny app's URL in a browser, enter a loan description, and click
**Predict**.

## Results

The final model reaches **0.90 accuracy** / **0.89 macro-F1** on the held-out
test set (400 loans), well above a most-frequent-class baseline (0.40
accuracy / 0.19 macro-F1). See `model_card.qmd` for the full breakdown,
including per-class metrics and the confusion matrix.

## Known limitations

- Trained on Uganda loans only; unlikely to generalize to other countries or
  languages.
- The 3-class sector grouping (`Agriculture` / `Business` / `Personal`) was
  chosen to balance class frequencies for this exercise and is not Kiva's
  original taxonomy.
- `deploy.py` and `app.py` are separate local processes and must both be
  running for the app to work — there's no supervisor process managing them
  in this local setup (a real deployment on Posit Connect would handle this).
