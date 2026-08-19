"""
Deploy step of the Kiva ML pipeline.

Loads the fitted pipeline from the local pins board and wraps it with
vetiver as a FastAPI app that serves predictions.

Run locally with:
    uvicorn deploy:app --reload --port 8080

Then visit http://127.0.0.1:8080/__docs__ for the interactive API docs, or
POST to /predict with a JSON body like:
    {"loan_amount": 500, "use": "to buy fertilizer for her coffee farm"}
"""
import pins
import vetiver

BOARD_PATH = "pins_board"
MODEL_PIN = "kiva_sector_model"

board = pins.board_folder(BOARD_PATH, allow_pickle_read=True)
model = vetiver.VetiverModel.from_pin(board, MODEL_PIN)

vetiver_api = vetiver.VetiverAPI(model, check_prototype=True)
app = vetiver_api.app
