# California Housing Estimator

A Streamlit interface for the registered California Housing champion model.
The model files are packaged with the app, so the running container does not
need access to the local MLflow database or tracking server.

## Run with Docker

Build and start the app from this folder:

```powershell
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501). Stop the app with
`Ctrl+C`, or run `docker compose down` in another terminal.

To build and run without Compose:

```powershell
docker build -t california-housing-ui .
docker run --rm -p 8501:8501 california-housing-ui
```

## Continuous integration

The repository-root `.github/workflows/docker-image.yml` workflow runs on every
push and pull request. It builds the Docker image, starts a container, checks
the Streamlit health endpoint and page, and confirms the packaged model can
make a finite prediction. It can also be started manually from the GitHub
Actions tab.

## Model input features

The form accepts median income, median house age, average rooms, average
bedrooms, population, average occupancy, latitude, and longitude. The displayed
estimate is the model prediction converted from units of $100,000 to US dollars.
It is a district-level estimate, not an appraisal of an individual property.

The image contains a snapshot of the `champion` model available when it was
packaged. After promoting a new champion, update the `model` directory from
that model's MLflow artifact and rebuild the image.
