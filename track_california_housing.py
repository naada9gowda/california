import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from mlflow.tracking import MlflowClient


REGISTERED_MODEL_NAME = "CaliforniaHousingLinearRegression"
MODEL_ALIAS = "champion"


def main():
    random_state = 42
    configurations = [
        ("standard_scaler", StandardScaler(), 0.2, True),
        ("min_max_scaler", MinMaxScaler(), 0.2, True),
        ("robust_scaler", RobustScaler(), 0.2, True),
        ("no_scaling", "passthrough", 0.2, True),
        ("standard_scaler", StandardScaler(), 0.3, True),
        ("standard_scaler", StandardScaler(), 0.2, False),
    ]

    data = fetch_california_housing()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment = mlflow.set_experiment(
        "california-housing-preprocessing-comparison"
    )
    client = MlflowClient()

    for preprocessing, scaler, test_size, fit_intercept in configurations:
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=test_size, random_state=random_state
        )
        model = make_pipeline(
            scaler,
            LinearRegression(fit_intercept=fit_intercept),
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        with mlflow.start_run(run_name=preprocessing):
            mlflow.log_param("model", "LinearRegression")
            mlflow.log_param("preprocessing", preprocessing)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("fit_intercept", fit_intercept)
            mlflow.log_param("random_state", random_state)
            mlflow.log_metric("mean_squared_error", mse)
            mlflow.log_metric("r2_score", r2)
            mlflow.sklearn.log_model(model, name="model")

        print(
            f"{preprocessing}, test_size={test_size}, "
            f"fit_intercept={fit_intercept}: MSE={mse:.3f}, R²={r2:.3f}"
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.r2_score DESC"],
    )
    columns = [
        "params.preprocessing",
        "params.test_size",
        "params.fit_intercept",
        "metrics.mean_squared_error",
        "metrics.r2_score",
    ]
    print("\nExperiment comparison (ranked by R²):")
    print(runs[columns].to_string(index=False))

    experiment_runs = client.search_runs([experiment.experiment_id])
    logged_models = client.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        max_results=10000,
    )
    model_by_run_id = {
        model.source_run_id: model
        for model in logged_models
        if model.source_run_id and model.status.value == "READY"
    }
    candidates = [
        run
        for run in experiment_runs
        if run.info.run_id in model_by_run_id
        and "mean_squared_error" in run.data.metrics
        and "r2_score" in run.data.metrics
    ]
    if not candidates:
        raise RuntimeError("No completed experiment runs with logged models were found.")

    best_run = min(
        candidates,
        key=lambda run: (
            -run.data.metrics["r2_score"],
            run.data.metrics["mean_squared_error"],
            -run.info.start_time,
        ),
    )
    best_logged_model = model_by_run_id[best_run.info.run_id]
    best_r2 = best_run.data.metrics["r2_score"]
    best_mse = best_run.data.metrics["mean_squared_error"]

    model_version = mlflow.register_model(
        model_uri=f"models:/{best_logged_model.model_id}",
        name=REGISTERED_MODEL_NAME,
        tags={
            "source_run_id": best_run.info.run_id,
            "selection_metric": "r2_score",
            "r2_score": str(best_r2),
            "mean_squared_error": str(best_mse),
        },
    )
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, MODEL_ALIAS, model_version.version
    )

    print(
        f"\nBest run: {best_run.info.run_id} "
        f"(preprocessing={best_run.data.params['preprocessing']}, "
        f"test_size={best_run.data.params['test_size']}, "
        f"fit_intercept={best_run.data.params['fit_intercept']})"
    )
    print(f"Best metrics: R²={best_r2:.3f}, MSE={best_mse:.3f}")
    print(
        f"Registered as {REGISTERED_MODEL_NAME} version {model_version.version}; "
        f"alias: {MODEL_ALIAS}"
    )


if __name__ == "__main__":
    main()
