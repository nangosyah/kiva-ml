"""
Shiny for Python app serving real-time sector predictions.

Loads the vetiver model pin directly and classifies a loan into
Agriculture / Business / Personal based on the borrower's stated use
of funds.

Run with:
    shiny run app.py

This app loads the model directly (no separate API process needed), which
keeps it deployable as a single piece of content on platforms like Posit
Connect Cloud that don't support a standalone FastAPI service alongside it.
If you do want the model served as its own API (e.g. for other clients to
call), see deploy.py.
"""
import pandas as pd
import pins
import vetiver
from shiny import App, reactive, render, ui

BOARD_PATH = "pins_board"
MODEL_PIN = "kiva_sector_model"

board = pins.board_folder(BOARD_PATH, allow_pickle_read=True)
model = vetiver.VetiverModel.from_pin(board, MODEL_PIN)

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
            result = model.model.predict(new_data)
        except Exception as e:
            return f"Prediction failed: {e}"
        return f"Predicted sector: {result[0]}"


app = App(app_ui, server)
