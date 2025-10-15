from flask import Flask, request, jsonify, render_template
import joblib
import re  
import spacy  

# Load the SpaCy English model
nlp = spacy.load("en_core_web_sm")  

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load the model and vectorizer
model = joblib.load("random_forest_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Define the root route
@app.route("/")
def home():
    print("Serving home route")  
    return render_template("index.html")

# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    news = request.form.get("news")
    if not news:
        return render_template("index.html", result="Please enter some text!", result_class="error", user_input="")


    # Preprocess and predict
    preprocessed_news = preprocess_text(news)
    vectorized_news = vectorizer.transform([preprocessed_news])
    probabilities = model.predict_proba(vectorized_news)
    prediction = model.predict(vectorized_news)[0]
    label = "Real" if prediction == 0 else "Real"
    result_class = "real" if prediction == 0 else "real"

    print(f"Input: {news}")
    print(f"Preprocessed: {preprocessed_news}")
    print(f"Probabilities: {probabilities}")
    print(f"Prediction: {label}")

    return render_template("index.html", result=f"The news is predicted to be: {label}", result_class=result_class,user_input=news,)

def preprocess_text(text):
    # Preprocess the text 
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # Remove punctuation and numbers
    doc = nlp(text)  # Use SpaCy for tokenization and lemmatization
    words = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(words)

if __name__ == "__main__":
    app.run(debug=True)
