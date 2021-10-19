import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import flask
from flask import request
import json

import pandas as pd

app = flask.Flask(__name__)
app.config["DEBUG"] = True

from flask import Flask
from flask_cors import CORS, cross_origin
import shutil
import copy
app = Flask(__name__)
CORS(app)
new_strings = []
datafiles=[]
from flask import jsonify,Response
# http://localhost:8000/
# print(new_strings)
@app.route('/', methods=['GET', 'POST'])
def home():
    maindata=request.data.decode('utf-8')
    import random
    import json

    import torch

    from model import NeuralNet
    from nltk_utils import bag_of_words, tokenize

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('intents.json', 'r') as json_data:
        intents = json.load(json_data)

    FILE = "data.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    bot_name = "Unibot"
    print("Let's chat! (type 'quit' to exit)")
    sentence = maindata
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                return Response("{'name':'ps'}",status=random.choice(intent['responses']))


    else:
        print(f"{bot_name}: I do not understand...")

    return Response("{'name':'ps'}",status='I do not understand...')
    # return Response("{'name':'ps'}",status=random.choice(intent['responses']))
    # data = {'name': 'nabin khadka'}
    # return jsonify(data)


app.debug = True
app.run()




