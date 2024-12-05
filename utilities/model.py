import pickle
import numpy as np

model = pickle.load(open('models/finalized_model.sav', 'rb'))
vc = pickle.load(open('models/vectorizer.pkl', 'rb'))

def get_prediction_score(input_text: str) -> float:
    features = vc.transform([input_text])
    decision_score = model.decision_function(features)
    min_decision_score = -1.5
    max_decision_score = 1.5
    scaled_score = 100 * (decision_score - min_decision_score) / (max_decision_score - min_decision_score)
    scaled_score_reversed = 100 - scaled_score
    scaled_score_reversed = np.clip(scaled_score_reversed, 0, 100)
    return scaled_score_reversed[0]

