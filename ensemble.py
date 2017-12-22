import pickle
from PIL import Image
from feature import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weaker = weak_classifier
        self.M = n_weakers_limit

    def is_good_enough(self,X,y):
        '''Optional'''
        y_pred = self.predict(X)
        y_pred.resize((len(y_pred),1))
        idx = np.where((y_pred-y)==0)   
        return len(idx[1])

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        n = X.shape[0]
        self.G = {}
        self.alpha = {}
        for i in range(self.M):
            self.G.setdefault(i)
            self.alpha.setdefault(i)
        self.sum=np.zeros(y.shape)    
        self.W=np.ones((n,1))/n 
        self.cnt=0 
        for i in range(self.M):
            w = self.W.flatten(1)
            self.G[i] = self.weaker.fit(X,y,sample_weight=w)
            e = self.G[i].score(X,y,sample_weight=w)
            if (1-e) > 0.5:
                break
            self.alpha[i] = 1/2*np.log((1-e)/e)
            h = self.G[i].predict(X)
            h.resize((n,1))
            #print('h',h)
            Z = np.multiply(self.W,np.exp(-self.alpha[i]*np.multiply(y,h)))
            #print('Z',Z)
            self.W = (Z/Z.sum())
            #print('W',self.W)
            self.cnt = i+1
            if self.is_good_enough(X,y) == 0:
                print(self.cnt,"weak classifiers is already good enough.")
                break

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        sum = np.zeros((X.shape[0],1))
        for i in range(self.cnt):
            t = -self.G[i].predict(X).flatten(1)*self.alpha[i] 
            t.resize((X.shape[0],1))
            sum = sum+t
        return sum


    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        y_pred = self.predict_scores(X)
        y_pred[y_pred>=threshold] = +1
        y_pred[y_pred<threshold] = -1
        return y_pred

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

