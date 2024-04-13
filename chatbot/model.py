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
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
import random
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning
import warnings


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
        self.intents = []  # Initialize self.intents

    def load_data(self):
        data = json.loads(open(self.intents_path, encoding='utf-8').read())
        self.intents = data['intents']  # Store the intents loaded from JSON
        for intent in self.intents:
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
        

    def check_intent_overlap(self, threshold=0.7):  # Adjust the threshold based on your needs
        vectorizer = TfidfVectorizer()
        phrases = [pattern for intent in self.intents for pattern in intent['patterns']]
        indices = {intent['tag']: i for i, intent in enumerate(self.intents)}
        vectorized_phrases = vectorizer.fit_transform(phrases)
        cos_sim = cosine_similarity(vectorized_phrases)

        overlap_suggestions = {}
        for i, intent_i in enumerate(self.intents):
            for j, intent_j in enumerate(self.intents):
                if i != j:
                    similarity = cos_sim[i, j]
                    if similarity > threshold:
                        print(f"Overlap detected between {intent_i['tag']} and {intent_j['tag']} with similarity {similarity:.2f}")
                        if intent_i['tag'] not in overlap_suggestions:
                            overlap_suggestions[intent_i['tag']] = []
                        overlap_suggestions[intent_i['tag']].append((intent_j['tag'], similarity))

        if overlap_suggestions:
            for intent, overlaps in overlap_suggestions.items():
                print(f"Consider merging '{intent}' with: {[tag for tag, _ in overlaps]}")
        else:
            print("No significant overlap detected among intents.")
    
    def suggest_mergers_or_splits(self):
        print("\nReviewing intents for potential mergers or splits...")
        for intent in self.intents:
            num_patterns = len(intent['patterns'])
            if num_patterns < 5:
                print(f"{intent['tag']} might need more examples or could be merged with a similar intent.")
            elif num_patterns > 15:
                print(f"{intent['tag']} might be too broad and could be split into more specific intents.")


    def analyze_intents(self):
        overlap_issues = []
        recommendation = {}
        for intent in self.intents:
            if len(intent['patterns']) < 3:
                recommendation[intent['tag']] = "Consider adding more examples."
            if len(intent['patterns']) > 10:
                recommendation[intent['tag']] = "Consider splitting this intent into more specific ones."
        if not recommendation:
            print("All intents seem well-configured.")
        else:
            for tag, advice in recommendation.items():
                print(f"Intent '{tag}': {advice}")

    def suggest_data_expansion(self):
        for intent in self.intents:
            if len(intent['patterns']) < 5:
                print(f"Intent '{intent['tag']}' has only {len(intent['patterns'])} examples. Consider expanding.")

    def update_intents_with_feedback(self, feedback_data):
        # Example feedback data format: {'intent': 'greeting', 'new_patterns': ['Hello there', 'Hi']}
        for intent in self.intents:
            if intent['tag'] == feedback_data['intent']:
                intent['patterns'].extend(feedback_data['new_patterns'])
                print(f"Updated patterns for intent '{intent['tag']}' with new examples.")
        self.save_data()

    def check_dataset_balance(self):
        counts = [len(intent['patterns']) for intent in self.intents]
        print(f"Total intents: {len(self.intents)}")
        print(f"Min patterns per intent: {min(counts)}")
        print(f"Max patterns per intent: {max(counts)}")
        print(f"Average patterns per intent: {np.mean(counts):.2f}")
        print(f"Standard deviation: {np.std(counts):.2f}")
        for intent in self.intents:
            print(f"{intent['tag']} has {len(intent['patterns'])} training phrases.")

    

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
        # Compute class weights for imbalanced datasets
        y_integers = np.argmax(train_y, axis=1)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers)
        d_class_weights = dict(enumerate(class_weights))
    
        # Define early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
        # Train the model
        history = self.model.fit(
            np.array(train_x), np.array(train_y),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            validation_split=0.2,
            callbacks=[early_stopping],
            class_weight=d_class_weights
        )
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
        # Load data, check for intent overlap, and balance the dataset
        self.load_data()
        self.check_intent_overlap()
        #self.check_dataset_balance()
        self.suggest_mergers_or_splits()
        self.analyze_intents()
        self.suggest_data_expansion()
        
        # Prepare the data for training and testing
        train_x, train_y = self.prepare_training_data()
        X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.1, random_state=42)
        
        # Build and train the model
        self.build_model(X_train, y_train)
        self.train(X_train, y_train)
        
        # Predict on the test set
        predictions = self.model.predict(np.array(X_test))
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        # Evaluate the model using all classes
        self.evaluate_model(predicted_classes, true_classes)
        
    def evaluate_model(self, predictions, true_labels):
        # Suppress undefined metric warnings
        warnings.filterwarnings('ignore', category=UndefinedMetricWarning) 
    
        # Calculate the additional metrics
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=1)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=1)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=1)
        
        # You must ensure that the labels parameter is set to all unique class labels from the training set.
        labels = sorted(np.unique(true_labels))  # Sort labels to maintain order
        # Update target_names to match the sorted labels
        target_names = [self.classes[i] for i in labels]
        
        # Generate the classification report using the sorted labels
        print(classification_report(true_labels, predictions, labels=labels, target_names=target_names, zero_division=1))
        
        # Print the additional metrics
        print(f'F1 Score: {f1:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')





if __name__ == "__main__":
    chatbot_model = ChatbotModel(intents_path='intents.json')
    chatbot_model.run()