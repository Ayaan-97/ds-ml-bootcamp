from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from pathlib import Path

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "breast_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def _get_feature(data, *names):
    for name in names:
        if name in data:
            return float(data[name])
    raise KeyError(f"Missing feature value for {names[0]}")


def _normalize_key(name):
    return name.strip().lower().replace(" ", "_")


def _build_feature_vector(data):
    provided = {
        _normalize_key(key): float(value)
        for key, value in data.items()
    }

    feature_names = list(getattr(scaler, "feature_names_in_", []))
    if not feature_names:
        return [[
            _get_feature(data, "radius_mean", "radius"),
            _get_feature(data, "texture_mean", "texture"),
            _get_feature(data, "perimeter_mean", "perimeter"),
            _get_feature(data, "area_mean", "area"),
            _get_feature(data, "smoothness_mean", "smoothness"),
            _get_feature(data, "compactness_mean", "compactness"),
            _get_feature(data, "concavity_mean", "concavity"),
            _get_feature(data, "concave_points_mean", "concave_points"),
            _get_feature(data, "symmetry_mean", "symmetry"),
            _get_feature(data, "fractal_dimension_mean", "fractal_dimension")
        ]]

    fallback_values = list(getattr(scaler, "mean_", [0.0] * len(feature_names)))
    values = []

    for index, feature_name in enumerate(feature_names):
        normalized_name = _normalize_key(str(feature_name))
        possible_keys = [normalized_name]

        if normalized_name.endswith("_mean"):
            possible_keys.append(normalized_name[:-5])

        if normalized_name == "concave_points_mean":
            possible_keys.append("concave_points")

        value = None
        for key in possible_keys:
            if key in provided:
                value = provided[key]
                break

        if value is None:
            value = float(fallback_values[index])

        values.append(value)

    return [values]

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    features = _build_feature_vector(data)

    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)[0]

    # Calculate confidence using decision function for SVC
    if hasattr(model, "decision_function"):
        # Get the distance from the decision boundary
        decision = model.decision_function(scaled_features)[0]
        # Convert to probability-like score using sigmoid function
        confidence = round(100 / (1 + __import__('math').exp(-decision)), 2)
    elif hasattr(model, "predict_proba"):
        probability = model.predict_proba(scaled_features)[0]
        confidence = round(max(probability) * 100, 2)
    else:
        confidence = 0.0

    result = "Malignant" if prediction == 1 else "Benign"

    return jsonify({
        "prediction": result,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)