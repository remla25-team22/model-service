from lib_ml.preprocess import clean_review as preprocess
import requests
import os, pickle, joblib, io
from flask import Flask, request, jsonify
from flasgger import Swagger
from sklearn.pipeline import Pipeline

MODEL_TAG = os.getenv("MODEL_TAG")
if not MODEL_TAG:
    raise RuntimeError("Missing MODEL_TAG env-var")

VECTOR_URL = f"https://github.com/remla25-team22/model-training/releases/download/{MODEL_TAG}/c1_BoW_Sentiment_Model.pkl"
CLF_URL    = f"https://github.com/remla25-team22/model-training/releases/download/{MODEL_TAG}/c2_Classifier_Sentiment_Model"

os.makedirs("model_cache", exist_ok=True)

resp = requests.get(VECTOR_URL); resp.raise_for_status()
vectorizer_path = "model_cache/vectorizer.pkl"
with open(vectorizer_path, "wb") as f:
    f.write(resp.content)
vectorizer = pickle.loads(resp.content)

resp = requests.get(CLF_URL); resp.raise_for_status()
classifier_path = "model_cache/classifier.pkl"
with open(classifier_path, "wb") as f:
    f.write(resp.content)
buf = io.BytesIO(resp.content)
classifier = joblib.load(buf)

model = Pipeline([
    ("vect", vectorizer),
    ("clf",  classifier)
])

app = Flask(__name__)
swagger = Swagger(app)

@app.route("/", methods=["POST"])
def predict():
    """
    Predict sentiment from a piece of text
    ---
    tags:
      - Sentiment
    parameters:
      - name: text
        in: body
        required: true
        schema:
          type: object
          properties:
            text:
              type: string
              description: The review text to classify
    responses:
      200:
        description: Prediction result
        schema:
          type: object
          properties:
            prediction:
              type: integer
              description: 0 = negative, 1 = positive
    """
    text = request.get_json().get("text", "")
    processed_text = preprocess(text)
    X_sparse = vectorizer.transform([processed_text])
    X_dense = X_sparse.toarray()
    pred = classifier.predict(X_dense)
    return jsonify({"prediction": int(pred)})

port_env = int(os.getenv("PORT", 8080))
app.run(host="0.0.0.0", port=port_env, debug=True)
