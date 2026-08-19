"""
Shiny for Python app serving real-time sector predictions.

Calls the local vetiver API (see deploy.py) to classify a loan into
Agriculture / Business / Personal based on the borrower's stated use
of funds.

Run the API first:
    uvicorn deploy:app --port 8080

Then run this app:
    shiny run app.py
"""
import pandas as pd
import requests
from shiny import App, reactive, render, ui
from vetiver import predict, vetiver_endpoint

ENDPOINT = vetiver_endpoint("http://127.0.0.1:8080/predict")

app_ui = ui.page_fluid(
    ui.h2("Kiva Loan Sector Predictor"),
    ui.input_text_area(
        "use",
        "Use of funds",
        value="to buy fertilizer to improve her soil for healthy coffee growing.",
        rows=4,
    ),
    ui.input_action_button("predict_btn", "Predict", class_="btn-primary"),
    ui.output_text_verbatim("prediction"),
)


def server(input, output, session):
    @render.text
    @reactive.event(input.predict_btn)
    def prediction():
        new_data = pd.DataFrame({"use": [input.use()]})
        try:
            result = predict(ENDPOINT, new_data)
        except requests.exceptions.ConnectionError:
            return (
                "Could not reach the model API at "
                f"{ENDPOINT}. Is it running? Start it with:\n"
                "  uvicorn deploy:app --port 8080"
            )
        except Exception as e:
            return f"Prediction failed: {e}"
        return f"Predicted sector: {result.iloc[0, 0]}"


app = App(app_ui, server)
