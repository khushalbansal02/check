import pickle
import numpy as np

model = pickle.load(open('models/finalized_model.sav', 'rb'))
vc = pickle.load(open('models/vectorizer.pkl', 'rb'))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_prediction(input_text: str):
    features = vc.transform([input_text])
    
    label = model.predict(features)[0]
    
    decision_score = model.decision_function(features)[0]
    
    probability = sigmoid(decision_score)
    probabilities = [1 - probability, probability]  
    label = int(label)
    if(label==1):
        label="Likely Fake"
    else:
        label="Likely True"
    probabilities = [float(prob) for prob in probabilities]
    print(label, probabilities)
    
    return label, probabilities

