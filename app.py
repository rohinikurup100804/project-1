from flask import Flask, request, render_template
import pickle

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news']
    vect_text = vectorizer.transform([news_text])
    prediction = model.predict(vect_text)
    result = 'FAKE' if prediction[0] == 0 else 'REAL'
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
