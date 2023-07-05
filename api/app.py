from flask import Flask, render_template, request
import neuralSentiment as ns

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    print(request.form)

    if request.method == "GET":
        return render_template('index.html')

    sentence = request.form['sentence']
    sentiment = ns.predict_sentiment(sentence)
    return render_template('index.html', sentiment=sentiment)
