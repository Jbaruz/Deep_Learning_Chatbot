import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from tensorflow.keras.models import load_model
import json
import pickle
import random

class ChatbotProcessor:
    def __init__(self, model_path='chatbot_model.h5', intents_path='intents.json', words_path='words.pkl', classes_path='classes.pkl'):
        self.lemmatizer = WordNetLemmatizer()
        self.model = load_model(model_path)
        self.intents = json.loads(open(intents_path, encoding='utf-8').read())
        self.words = pickle.load(open(words_path, 'rb'))
        self.classes = pickle.load(open(classes_path, 'rb'))

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(self, sentence, show_details=True):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print(f"found in bag: {w}")
        return np.array(bag)

    def predict_class(self, sentence):
        p = self.bow(sentence, show_details=False)
        res = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list

    def get_response(self, ints):
        tag = ints[0]['intent']
        list_of_intents = self.intents['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        else:
            result = "Sorry, I do not understand. Can you rephrase?"
        return result

    def chatbot_response(self, msg):
        ints = self.predict_class(msg)
        res = self.get_response(ints)
        return res

      
if __name__ == "__main__":
    chatbot_processor = ChatbotProcessor()
    print("Chatbot is running. Type 'bye' to exit.")
    while True:
        message = input("You: ")
        if message.lower() == "bye":
            print("Bot: Goodbye!")
            break
        response = chatbot_processor.chatbot_response(message)
        print("Bot:", response)

