# KIVA-ML

An end-to-end Python ML pipeline, built as a tutorial example, using a single-language stack: `pins` for data and model versioning, `vetiver` for model deployment, and Shiny for Python for the front end.

**Task:** classify a Kiva Uganda microloan into one of three sectors (`Agriculture`, `Business`, `Personal`) from the borrower's free-text description of what the loan will be used for.

## Pipeline stages

```
kiva_extract.py  -->  transform.py  -->  train.py  -->  evaluate.py
                                              |
                                              v
                                         deploy.py  -->  app.py
```

| Stage     | Script              | What it does                                                                                                                                                                                                                                                               |
| --------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Extract   | `kiva_extract.py` | Pulls Uganda loan listings from the public Kiva API (`api.kivaws.org`) and writes the raw data to `kiva_uganda.csv`.                                                                                                                                                   |
| Transform | `transform.py`    | Tidies the raw data: engineers`description_length`, merges rare sectors (<20 obs) into `Other`, then groups sectors into 3 balanced classes. Writes the result to the `kiva_tidy_loans` pin on a local pins board (`pins_board/`).                                 |
| Train     | `train.py`        | Splits into train/test (80/20, stratified), cross-validates a TF-IDF +`LogisticRegression` pipeline, fits it on the training split, and saves it as a `vetiver` model pin (`kiva_sector_model`). Also pins the untouched held-out test split (`kiva_sector_test`). |
| Evaluate  | `evaluate.py`     | One-time final evaluation of the pinned model against the held-out test pin. Run this only after model development is complete.                                                                                                                                            |
| Deploy    | `deploy.py`       | Loads the model pin and serves it as a local FastAPI app via`vetiver.VetiverAPI`, exposing a `/predict` endpoint.                                                                                                                                                      |
| App       | `app.py`          | A Shiny for Python app with a text box for the loan description; calls the local`/predict` endpoint and displays the predicted sector.                                                                                                                                   |

Model documentation lives in [`model_card.qmd`](model_card.qmd), covering intended use, metrics, and known limitations.

## Running it

Install dependencies. Two options depending on your workflow:

```bash
# using pip (also what Posit Connect will use to build its own environment)
pip install -r requirements.txt

# or using uv for local development (creates and manages .venv for you)
uv sync
```

Both `requirements.txt` and `pyproject.toml`/`uv.lock` pin the same exact
versions used to build and evaluate this project (Python 3.9.16,
`scikit-learn==1.6.1`, etc.). `uv.lock` additionally pins the full transitive
dependency graph (157 packages) for fully reproducible local installs; `uv
lock` regenerates it if you ever change `pyproject.toml`.

Run the pipeline stages in order:

```bash
python kiva_extract.py   # writes kiva_uganda.csv
python transform.py      # writes kiva_tidy_loans pin
python train.py          # writes kiva_sector_model + kiva_sector_test pins
python evaluate.py       # one-time test-set evaluation, prints metrics
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

The final model reaches **0.90 accuracy** / **0.89 macro-F1** on the held-out test set (400 loans), well above a most-frequent-class baseline (0.40 accuracy / 0.19 macro-F1). See `model_card.qmd` for the full breakdown, including per-class metrics and the confusion matrix.

## Deploying to Posit Connect

`requirements.txt` pins exact versions of every runtime dependency, matching the environment (Python 3.9.16) that trained and pinned the model. This matters more than it might look: scikit-learn pickles are version sensitive, and an unpinned or mismatched scikit-learn build will fail to unpickle the model pin.

The app and the model API are two separate pieces of content, so deploy them separately with [`rsconnect-python`](https://docs.posit.co/rsconnect-python/) (already included transitively via `vetiver`):

```bash
# the vetiver model API (deploy.py)
rsconnect deploy fastapi --server <CONNECT_URL> --api-key <API_KEY> .

# the Shiny app (app.py)
rsconnect deploy shiny --server <CONNECT_URL> --api-key <API_KEY> .
```

A couple of things to check before deploying:

- Update `ENDPOINT` in `app.py` to point at the deployed API's Connect URL instead of `http://127.0.0.1:8080/predict`.
- `deploy.py` currently reads the model from a local `pins_board/` folder, which gets bundled with the content automatically. For a setup where the model can be updated without redeploying the app, consider switching to `pins.board_connect()` so the pin lives on Connect independently of the app's code bundle.
- Confirm the Connect server has a Python 3.9 runtime configured; if not, retrain and re-pin the model under whichever Python/scikit-learn version Connect does support, then update `requirements.txt` to match.

## Limitations

- Trained on Uganda loans only; unlikely to generalize to other countries or languages.
- The 3-class sector grouping (`Agriculture` / `Business` / `Personal`) was chosen to balance class frequencies for this exercise and is not Kiva's original taxonomy.
- `deploy.py` and `app.py` are separate local processes and must both be running for the app to work, there's no supervisor process managing them in this local setup (a real deployment on Posit Connect would handle this).
