import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from itertools import product
from tensorflow.examples.tutorials.mnist import input_data


class RFLayer_RAND(object):
    def __init__(self, n_estimators, classifier=True, md=None, mss=10):
        self.n_estimators = n_estimators
        self.max_depth = md
        self.min_samples_split = mss
        self.classifier = classifier

    def fit(self, X_train, y_train, kfold=5, k=2, n_jobs=-1): # kfold = 5 yields 80/20 split, k will be the number of times we run validation
        if kfold > 1:
            kf = KFold(kfold, shuffle=True)
        else:
            raise ValueError('Need to pass kfold something greater than 1 so can do cross validation')

        models = []
        best_score = 0
        best_ind = 0
        count = 0

        # split training data into training and estimating sets via quasi kfold validation routine
        for tr_ind, est_ind in kf.split(X_train, y_train):
            # instantiate the layer of decision trees
            models.append(RandomForestClassifier(self.n_estimators, criterion='entropy', max_depth=self.max_depth,
                                                 min_samples_split=self.min_samples_split,
                                                 n_jobs=n_jobs))
            for tree in models[count].estimators_: # make half of the trees completely random Decision Trees
                if np.random.rand() <= .5:
                    tree.splitter = 'random'


            # get the split of the training data
            X_tr, y_tr = X_train[tr_ind,:], y_train[tr_ind]
            # train the layer on this split
            models[count].fit(X_tr, y_tr)
            X_tr, y_tr = 0, 0

            # check accuracy on the estimation set
            X_est, y_est = X_train[est_ind,:], y_train[est_ind]
            y_pred = models[count].predict(X_est)
            acc_score = accuracy_score(y_pred, y_est)
            X_est, y_est = 0, 0 # memory
            y_pred = 0 # memory

            if acc_score > best_score: # with k > 1 we compare to see which is best layer trained
                best_score = acc_score
                best_ind = count
            count += 1
            if count >= k:
                break

        # save the best layer
        self.L = models[best_ind]
        self.n_classes = self.L.n_classes_
        self.val_score = best_score

    def predict(self, X_test):
        return self.L.predict(X_test)

    def push_thru_data(self, X):
        n_samples, dim_data = X.shape
        X_push = np.empty((n_samples, self.n_estimators*self.n_classes))
        # push the data X through this layer
        i = 0
        for tree in self.L.estimators_:
            if self.classifier:
                X_push[:,i*self.n_classes:(i+1)*self.n_classes] = tree.predict_proba(X).astype('float32')
            i += 1
        return X_push
