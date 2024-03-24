from flask import render_template, request, jsonify
from app import app
from chatbot.processing import ChatbotProcessor

# Initialize ChatbotProcessor with paths to your model, intents, words, and classes
# Ensure these paths are correct relative to your Flask application's start location
chatbot_processor = ChatbotProcessor(
    model_path='chatbot/chatbot_model.h5',
    intents_path='chatbot/intents.json',
    words_path='chatbot/words.pkl',
    classes_path='chatbot/classes.pkl'
)

@app.route('/')
def index():
    # Serve the main page
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    # Extract the question from the request
    data = request.json
    question = data.get('question')

    # Use ChatbotProcessor to get the response
    response = chatbot_processor.chatbot_response(question)
    
    # Return the response as JSON
    return jsonify({"response": response})
