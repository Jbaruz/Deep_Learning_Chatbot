import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.optimizers.schedules import ExponentialDecay
import random
import os

print("Current Working Directory:", os.getcwd())

class ChatbotModel:
    def __init__(self, intents_path='intents.json'):
        self.intents_path = intents_path
        self.lemmatizer = WordNetLemmatizer()
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?', '!']
        self.model = None

    def load_data(self):
        intents = json.loads(open(self.intents_path, encoding='utf-8').read())
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                w = nltk.word_tokenize(pattern)
                self.words.extend(w)
                self.documents.append((w, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))

        # Save words and classes to pkl files
        pickle.dump(self.words, open('words.pkl', 'wb'))
        pickle.dump(self.classes, open('classes.pkl', 'wb'))

    def prepare_training_data(self):
        training = []
        output_empty = [0] * len(self.classes)
        for doc in self.documents:
            bag = []
            pattern_words = [self.lemmatizer.lemmatize(word.lower()) for word in doc[0]]
            bag = [1 if w in pattern_words else 0 for w in self.words]
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training, dtype=object)
        train_x = list(training[:,0])
        train_y = list(training[:,1])
        return train_x, train_y

    def build_model(self, train_x, train_y):
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(train_y[0]), activation='softmax'))

        # Learning rate schedule
        lr_schedule = ExponentialDecay(initial_learning_rate=0.01, decay_steps=100000, decay_rate=0.96, staircase=True)
        
        sgd = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    def train(self, train_x, train_y, epochs=200, batch_size=5):
        history = self.model.fit(np.array(train_x), np.array(train_y), epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)
        
        # Save model
        self.model.save('chatbot_model.h5')

        # Plotting training history
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    def run(self):
        self.load_data()
        train_x, train_y = self.prepare_training_data()
        self.build_model(train_x, train_y)
        self.train(train_x, train_y)

if __name__ == "__main__":
    chatbot_model = ChatbotModel(intents_path='intents.json')
    chatbot_model.run()
