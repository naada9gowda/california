import math
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from flask import Flask, jsonify, request


MODEL_NAME = "CaliforniaHousingLinearRegression"
MODEL_ALIAS = "champion"
FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def create_app():
    app = Flask(__name__)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri is None:
        database_path = (Path(__file__).parent / "mlflow.db").resolve().as_posix()
        tracking_uri = f"sqlite:///{database_path}"
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

    @app.get("/")
    def index():
        return jsonify(
            service="California Housing prediction API",
            endpoints={
                "health": "GET /health",
                "predict": "POST /predict",
            },
            prediction_request={
                "features": {name: "number" for name in FEATURE_NAMES}
            },
        )

    @app.get("/health")
    def health():
        return jsonify(
            status="ok",
            registered_model=MODEL_NAME,
            model_alias=MODEL_ALIAS,
        )

    @app.post("/predict")
    def predict():
        if not request.is_json:
            return jsonify(error="Content-Type must be application/json."), 400

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Request body must be a JSON object."), 400

        features = payload.get("features")
        if not isinstance(features, dict):
            return jsonify(error="'features' must be an object of feature names and values."), 400

        missing = [name for name in FEATURE_NAMES if name not in features]
        unknown = [name for name in features if name not in FEATURE_NAMES]
        if missing or unknown:
            return jsonify(
                error="Provide exactly the expected California Housing features.",
                missing_features=missing,
                unknown_features=unknown,
                expected_features=FEATURE_NAMES,
            ), 400

        values = [features[name] for name in FEATURE_NAMES]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            return jsonify(error="All feature values must be finite numbers."), 400

        prediction = float(model.predict([values])[0])
        return jsonify(
            prediction=prediction,
            prediction_units="hundreds of thousands of US dollars",
            registered_model=MODEL_NAME,
            model_alias=MODEL_ALIAS,
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5001")))
