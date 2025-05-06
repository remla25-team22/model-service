from lib_ml import preprocess
import requests
import os, pickle, joblib, requests, io
from flask import Flask, request, jsonify
from flasgger import Swagger
from sklearn.pipeline import Pipeline


MODEL_TAG = os.getenv("MODEL_TAG")
if not MODEL_TAG:
    raise RuntimeError("Missing MODEL_TAG env-var")


VECTOR_URL = f"https://github.com/remla25-team22/model-training/releases/download/{MODEL_TAG}/c1_BoW_Sentiment_Model.pkl"
CLF_URL =    f"https://github.com/remla25-team22/model-training/releases/download/{MODEL_TAG}/c2_Classifier_Sentiment_Model"


resp = requests.get(VECTOR_URL); resp.raise_for_status()
vectorizer = pickle.loads(resp.content)


# Download & load the classifier
resp = requests.get(CLF_URL); resp.raise_for_status()
# joblib.dump by default writes a binary .pkl—so:
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
    X_sparse = vectorizer.transform([text])
    X_dense = X_sparse.toarray()
    pred = classifier.predict(X_dense)
    return jsonify({"prediction": int(pred)})


app.run(host="0.0.0.0", port=8080, debug=True)
