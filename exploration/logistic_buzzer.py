# Jordan Boyd-Graber
# 2023
#
# Buzzer using Logistic Regression

import pickle

from sklearn.linear_model import LogisticRegression

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

        pickle.dump(self._features[1].cached_pages, open("wiki_pages.pkl", "wb"))

    def load(self):
        Buzzer.load(self)
        with open("%s.model.pkl" % self.filename, 'rb') as infile:
            self._classifier = pickle.load(infile)
