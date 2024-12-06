from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  
from utilities.chatbot import get_response  
from utilities.model import get_prediction_score 
from utilities.embedding import search_and_store_embeddings
import os 

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/misinfo_chatbot', methods=['POST'])
def misinfo_chatbot():
    try:
        data = request.get_json()
        post_title = data.get('post_title', '')
        post_content = data.get('post_content', '')
        user_query = data.get('query', '')
        conversation_history = data.get('conversation_history', '')

        if not post_content or not user_query:
            return jsonify({'error': 'Post content and user query are required.'}), 400

        response = get_response(post_title, post_content, user_query, conversation_history)
        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
      
@app.route('/predict_score', methods=['POST'])
def predict_score():
    try:
        data = request.get_json()
        post_title = data.get('post_title', '')
        post_content = data.get('post_content', '')
        input_text= post_title+ " "+ post_content

        if not input_text:
            return jsonify({'error': 'Text input is required for prediction.'}), 400

        score = get_prediction_score(input_text)  
        return jsonify({'prediction_score': score})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/store_embeddings', methods=['POST'])
def store_embeddings():
    tags = request.args.get('tags', '').split(',')

    if not tags :
        return jsonify({"error": "Tags required"}), 400

    try:
        message = search_and_store_embeddings(tags)
        return jsonify({"message": message}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
  app.run(port=5000)
