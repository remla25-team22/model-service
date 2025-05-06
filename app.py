from lib_ml import preprocess
import requests
import os, pickle, joblib, requests, io
from flask import Flask, request, jsonify
from flasgger import Swagger
from sklearn.pipeline import Pipeline


# Hard-coded model URL
VECTOR_URL = "https://github.com/remla25-team22/model-training/releases/download/v1.0.1/c1_BoW_Sentiment_Model.pkl"
CLF_URL = "https://github.com/remla25-team22/model-training/releases/download/v1.0.1/c2_Classifier_Sentiment_Model"

resp = requests.get(VECTOR_URL); resp.raise_for_status()
vectorizer = pickle.loads(resp.content)


# Download & load the classifier
resp = requests.get(CLF_URL); resp.raise_for_status()
# joblib.dump by default writes a binary .pkl—so:
buf = io.BytesIO(resp.content)
classifier = joblib.load(buf)

# Optional: build a small pipeline so you can still call .predict on one object
model = Pipeline([
  ("vect", vectorizer),
  ("clf",  classifier)
])

# Serve predictions
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
    # 1) vectorize → sparse
    X_sparse = vectorizer.transform([text])
    # 2) convert to dense exactly like your training did
    X_dense = X_sparse.toarray()
    # 3) predict
    print("The input is ", len(X_dense[0]))
    pred = classifier.predict(X_dense)
    print("the model output is ", pred)
    return jsonify({"prediction": int(pred)})


app.run(host="0.0.0.0", port=8080, debug=True)
