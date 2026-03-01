from flask import Flask, render_template, request, redirect
from prediction_pipeline import preprocessing, vectorizer, get_prediction
from logger import logging

app = Flask(__name__)
logging.info("Flask server started..")

# Define global variables here
reviews = []
positive = 0
negative = 0


@app.route("/")
def index():
    data = {
        'reviews': reviews,
        'positive': positive,
        'negative': negative
    }
    logging.info("open home page..)")
    return render_template('index.html', data=data)


@app.route("/", methods=['POST'])
def post():
    global positive, negative, reviews  #Tell Python to use global variables

    text = request.form['text']
    logging.info(f"Text:{text}")

    preprocessed_txt = preprocessing(text)
    vectorized_txt = vectorizer(preprocessed_txt)
    prediction = get_prediction(vectorized_txt)
    logging.info(f"Prediction:{prediction}")

    if prediction == 'negative':
        negative += 1
    else:
        positive += 1

    reviews.insert(0, text)

    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)