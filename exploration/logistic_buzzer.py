# Jordan Boyd-Graber
# 2023
#
# Buzzer using Logistic Regression

import pickle
import sklearn_crfsuite
import collections
from sklearn.linear_model import LogisticRegression
from sklearn_crfsuite import CRF

from buzzer import Buzzer

class LogisticBuzzer(Buzzer):
    """
    Logistic regression classifier to predict whether a buzz is correct or not.
    """

    def train(self):
        X = Buzzer.train(self)
        
        self._classifier = LogisticRegression()
        self._classifier.fit(X, self._correct)

    def save(self):
        Buzzer.save(self)
        with open("%s.model.pkl" % self.filename, 'wb') as outfile:
            pickle.dump(self._classifier, outfile)

    def load(self):
        Buzzer.load(self)
        with open("%s.model.pkl" % self.filename, 'rb') as infile:
            self._classifier = pickle.load(infile)

# Change the buzzer to use the guess history and feed it into a crf-suite model.
class BuzzBetter(LogisticBuzzer):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._guess_histories = defaultdict(list)

    def featurize(self, question, run_text, guess_history, guesses=None):
        # Call parent class's featurize
        guess, features = super().featurize(question, run_text, guess_history, guesses)

        # Add current guess to guess history for current question
        qid = question["qanta_id"]
        self._guess_histories[qid].append(guess)

        return guess, features
    
    def build_sequence_features(self):
        """
        Convert guess histories into sequence features.
        """
        X_train = []
        y_train = []

        for qid, guesses in self._guess_histories.items():
            sequence_features = []
            labels = []

            for guess in guesses:
                # For simplicity, we just use the guess as a feature. 
                # You can expand on this to include more sophisticated features.
                sequence_features.append({'guess': guess})
                labels.append(rough_compare(guess, self._answers[qid]) * 1)

            X_train.append(sequence_features)
            y_train.append(labels)

        return X_train, y_train
    
    def train(self):
        """
        Override the train method to use CRF-suite.
        """
        X, y = self.build_sequence_features()

        self._classifier = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        self._classifier.fit(X, y)