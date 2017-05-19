### script to make the windowing files for the whole MNIST dataset, along with building the random forests for MGS step and pushing all the training and testing data thru


import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import pickle
import sys
import os
from sklearn.metrics import accuracy_score
from itertools import product
from keras.datasets import mnist
from functions import windowing_tr, windowing_push, push_thru_MGS_windowing_sep



# get the MNIST data
(X_tr, y_tr), (X_t, y_t) = mnist.load_data()
X_tr = X_tr.reshape(60000, 784)
X_t, y_t = 0, 0 # memory efficiency


# train the Multi-Grained Scanning Random Forests, Windowing
n_trees = 30
n_samples = X_tr.shape[0] # number of training samples
MGS_Forests = []
w = int(sys.argv[1])
test_name = str(w)+ '_wind'
foldername = './'+test_name + '/'
os.mkdir(foldername)

print w
# do multi-grained scanning - windowing. save files
Xw, yw = windowing_tr(X_tr,y_tr,  w_size=w)
y_tr = 0 # memory

# build the Random Forests for MGS for this window size
forest1 = RandomForestClassifier(n_trees, max_depth=None, min_samples_split=20, n_jobs=-1)
forest2 = ExtraTreesClassifier(n_trees, max_depth=None, min_samples_split=20, n_jobs=-1)

# fit the forests to windowed data
forest1.fit(Xw, yw)
forest2.fit(Xw, yw)
Xw, yw = 0,0 # free up memory

# append forests to list, pickle to save
MGS_Forests.append(forest1)
MGS_Forests.append(forest2)
forest1, forest2 = 0, 0 # memory

# push training data thru the mgs random forests
X_tr_thru = push_thru_MGS_windowing_sep(X_tr, MGS_Forests, w)
np.save(foldername+'X_tr_thru_'+test_name, X_tr_thru)
X_tr_thru = 0 # memory

# push the testing data thru
# get the MNIST data
(X_tr, y_tr), (X_t, y_t) = mnist.load_data()
X_tr, y_tr = 0, 0 # for memory
X_t = X_t.reshape(10000, 784)
y_t = 0 # for memory

# push the testing data thru MGS random forests
X_t_thru = push_thru_MGS_windowing_sep(X_t, MGS_Forests, w)

np.save(foldername+'X_t_thru_'+test_name, X_t_thru)
