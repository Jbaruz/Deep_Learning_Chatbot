# model.py
# -----------------------------------------------------------------------------------
# Original Author: Juan
# Purpose: This module defines the neural network architecture and training procedures for a chatbot system.
# Last Modified Date:  13/04/2024
# -----------------------------------------------------------------------------------

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

"""
A chatbot model capable of understanding and responding based on a predefined set of intents.

Attributes:
    intents_path (str): Path to the JSON file containing the intents.
    lemmatizer (WordNetLemmatizer): A tool for reducing words to their base form.
    words (list): List of tokenized and lemmatized unique words.
    classes (list): List of unique intent tags.
    documents (list): List of tuples pairing patterns with their corresponding tags.
    ignore_words (list): Words to be ignored during data processing.
    model (Sequential): The compiled Keras model.
"""
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
        """
        Loads and preprocesses the intent data from the JSON file specified by intents_path.
        Side effects:
            Modifies self.words, self.classes, self.documents by populating them with processed data.
            Saves the processed words and classes as pickle files.
        """
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
        """
        Prepares training data by converting each pattern into a "bag of words".
        Returns:
            Tuple[np.array, np.array]: Training inputs and outputs suitable for neural network training.
        """
        training = []
        output_empty = [0] * len(self.classes)
        # Initialize bag of words for each pattern.
        for doc in self.documents:
            bag = []
            pattern_words = [self.lemmatizer.lemmatize(word.lower()) for word in doc[0]]
            bag = [1 if w in pattern_words else 0 for w in self.words]
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])

        # Shuffle the training data to prevent bias.
        random.shuffle(training)
        # Convert training data to numpy array for efficient processing.
        training = np.array(training, dtype=object)
        # Training inputs
        train_x = list(training[:,0])
        # Training outputs
        train_y = list(training[:,1])
        return train_x, train_y

    def build_model(self, train_x, train_y):
        """
        Builds the neural network model.
        Parameters:
            train_x (list): Training data inputs.
            train_y (list): Training data outputs.
        """

        # Initialize the sequential model
        '''  
        Probability Output: 
        Softmax transforms the outputs into probabilities. Each output number represents the likelihood of the input belonging to a certain class, with all probabilities adding up to 1.
        
        Multi-class Classification: 
        Softmax is ideal for problems where each input must be categorized into one, and only one, of many possible classes. It’s commonly used in chatbots to determine which category a user's question fits into.
        
        Clear Decision Boundaries: 
        Softmax helps the model make clear decisions by defining distinct boundaries between classes. It provides probabilities that help the model be more certain about which class to assign to an input.
        
        Works with Cross-Entropy Loss: 
        It pairs well with categorical cross-entropy loss, a common way to measure how well the model’s predictions match the actual labels. This is crucial for training models to be accurate.
        Softmax is effective for ensuring that the model can confidently and accurately categorize inputs, which is essential for applications like chatbots where precise understanding is key.  
        '''
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))# Add first dense layer.
        self.model.add(Dropout(0.5)) # Add dropout to prevent overfitting.
        self.model.add(Dense(64, activation='relu')) # Add second dense layer
        self.model.add(Dropout(0.5)) # Add dropout to prevent overfitting.
        self.model.add(Dense(len(train_y[0]), activation='softmax')) # Add output layer

        # Learning rate schedule
        lr_schedule = ExponentialDecay(initial_learning_rate=0.01, decay_steps=100000, decay_rate=0.96, staircase=True)
        
        sgd = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
        # loss function 'categorical_crossentropy' is ideal for training classification models because it effectively drives the model
        # towards more accurate and confident predictions by aligning the predicted probabilities with the true labels.
        # It's particularly useful when combined with the Softmax function, as it helps to ensure that the model's
        # outputs can be interpreted directly as probabilities of each class, providing a clear and actionable output
        # for decision-making processes.
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    def train(self, train_x, train_y, epochs=100, batch_size=5):
        """
        Trains the model with the provided training data.
        Parameters:
            train_x (list): Training data inputs.
            train_y (list): Training data outputs.
            epochs (int): Number of training epochs.
            batch_size (int): Size of each training batch.
        """
        history = self.model.fit(np.array(train_x), np.array(train_y), epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)
        
        # Save model
        self.model.save('chatbot_model.h5')

        # Plotting training history
        # Plot training and validation accuracy.
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

        # Plot training and validation loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss']) #
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