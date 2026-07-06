# Car MPG Predictor API

A FastAPI web service that predicts a car's fuel efficiency (miles per gallon) from seven vehicle specifications, using a Lasso regression model with polynomial features.

## Overview

The prediction pipeline is:

`StandardScaler` → `PolynomialFeatures` → `Lasso Regression`

The model, scaler, and polynomial transformer are already trained and saved as `.pkl` files, so no training step is required — the API loads them directly and serves predictions.

## Input features

The model expects these seven features, in this order:

| Feature        | Description                                | Example |
|----------------|--------------------------------------------|---------|
| `cylinders`    | Number of engine cylinders                 | 4       |
| `displacement` | Engine displacement (cubic inches)         | 150.0   |
| `horsepower`   | Gross engine horsepower                    | 100.0   |
| `weight`       | Vehicle weight (pounds)                     | 3000.0  |
| `acceleration` | Time to accelerate 0–60 mph (seconds)      | 15.0    |
| `model_year`   | Model year (70–82, meaning 1970–1982)      | 80      |
| `origin`       | Region of manufacture (1=USA, 2=Europe, 3=Japan) | 1  |

## Project structure

```
MPG_predicter/
├── main.py              # FastAPI app that serves predictions
├── lasso_model.pkl      # Trained Lasso regression model
├── std_scaler.pkl       # Fitted StandardScaler
├── poly_feats.pkl       # Fitted PolynomialFeatures transformer
├── requirements.txt     # Python dependencies
└── README.md
```

## Requirements

- Python 3.10 or newer
- pip

## Setup

1. **Open a terminal** in the project folder (`MPG_predicter`).

2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies:**

   ```bash
   pip install fastapi uvicorn scikit-learn joblib numpy
   ```

   Make sure the three `.pkl` files are in the same folder as `main.py`.

## Usage

### Start the API

```bash
uvicorn main:app --reload
```

The server runs at `http://127.0.0.1:8000`.

> **Note:** The command is `uvicorn main:app`, where `main` is the filename (`main.py`) and `app` is the FastAPI object inside it.

### Test it

Open the interactive documentation in your browser:

```
http://127.0.0.1:8000/docs
```

From there you can send test requests directly from the browser.

## API reference

### `GET /`

Returns a simple welcome message confirming the API is running.

### `POST /predict`

Predicts the MPG of a car.

**Request body:**

```json
{
  "cylinders": 4,
  "displacement": 150.0,
  "horsepower": 100.0,
  "weight": 3000.0,
  "acceleration": 15.0,
  "model_year": 80,
  "origin": 1
}
```

**Response:**

```json
{
  "predicted_mpg": 24.55
}
```

**Example with curl:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"cylinders": 4, "displacement": 150.0, "horsepower": 100.0, "weight": 3000.0, "acceleration": 15.0, "model_year": 80, "origin": 1}'
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `Error loading ASGI app. Could not import module "app"` | Wrong filename in command | Use `uvicorn main:app`, not `app:app` |
| `Attribute "app" not found in module "main"` | `main.py` is empty or missing the `app` object | Check the file has content with `cat main.py` |
| `[Errno 98] Address already in use` | Port 8000 already taken by another server | Run `pkill -f uvicorn`, then start again |
| `FileNotFoundError: lasso_model.pkl` | A `.pkl` file is missing from the folder | Ensure all three `.pkl` files sit next to `main.py` |
| `InconsistentVersionWarning` | `.pkl` files trained with a different scikit-learn version | Harmless; predictions still work. To silence it, run `pip install scikit-learn==1.6.1` |

## Notes

- The API field is named `model_year` (with an underscore) because a space is not valid in a JSON field name. This does not affect predictions, since features are passed to the model by position, not by name.
- No training script is included because the model is already trained and saved in the `.pkl` files.

## Dependencies

- fastapi
- uvicorn
- scikit-learn
- joblib
- numpy

## License

This project is for educational purposes.