import pickle, json
from tensorflow import keras
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

with open('./savedModels/tokenizer.json') as f:
    data = json.load(f)
    tokenizer = keras.preprocessing.text.tokenizer_from_json(data)

# with open('./savedModels/tokenizer.pkl', 'rb') as handle:
#     tokenizer = pickle.load(handle)

model = keras.models.load_model('./savedModels/model.h5')

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods = ['POST'])
@cross_origin()
def basic():
    text = request.json['text']
    print(text)
    X_test = tokenizer.texts_to_sequences([text])
    padded_text = keras.preprocessing.sequence.pad_sequences(X_test, padding='post', maxlen=50)
    y_pred = model.predict(padded_text)
    if y_pred <= 0.75:
        response = 'Not a Spam'
    else:
        response = 'spam'
    return jsonify({'response' : response})

if __name__ == '__main__':
    app.run(debug = True)