import numpy as np
from sklearn.metrics import accuracy_score
from keras.datasets import mnist
from structure import RFLayer_RAND


print 'Loading in training data:'
# This is the data read in from previous layers, output of the function concat_mgs_output
(X_tr, y_tr), (Xtest, ytest) = mnist.load_data()
X_tr, Xtest, ytest = 0, 0, 0
X_curr = np.load('./X_mgsout.npy')


print
print 'RF Layer training:'

# parameters for the building of the next layers
n = 2000 # num trees in each layer
min_gain = 0.01
verbose = True
max_layers = 5
md = None
mss = 10
n_jobs = -1

# dictionary where layers of decision trees will be stored
Layers = {}

# train the FTDRF layers on MGS data loaded in
prev_score = -1.0 # instantiate prev_score
for i in xrange(max_layers):
    print X_curr.shape
    RFL = RFLayer_RAND(n, md=md, mss=mss)
    RFL.fit(X_curr, y_tr, 5, 1, n_jobs)
    Layers[i] = RFL

    # if verbose, print out the estimation accuracy for this layer
    if verbose:
        print 'Layer ' + str(i+1)
        print 'acc: ' + str(RFL.val_score)


    # check to see if we have improved enough going one more layer
    rel_gain = (RFL.val_score - prev_score)/float(abs(prev_score))
    if rel_gain < min_gain or RFL.val_score == 1.0:
        print 'Converged! Stopping building layers'
        print
        break
    prev_score = RFL.val_score

    # if moving on to another level, push the data through
    X_curr = RFL.push_thru_data(X_curr)
    print 'Going to another layer'
    print

X_curr, y_sp = 0, 0 # memory

# load in testing data, free up memory of the training data
print 'Loading in testing data'
(X_tr, y_tr),(X_t, y_t) = mnist.load_data()
X_tr, y_tr, X_t = 0, 0, 0
y_t = y_t.astype('uint8')
X_t_curr = np.load('./X_t_mgsout.npy')


# push test data thru FTDRF layers
for i in xrange(len(Layers.keys())-1):
    X_t_curr = Layers[i].push_thru_data(X_t_curr)
last = len(Layers.keys())-1
y_pred = Layers[last].predict(X_t_curr)


print
print 'Statistics:'
print 'The accuracy was:'
print accuracy_score(y_pred, y_t)
print 'Params:'
print 'num_tres in each layer = ' + str(n)
print 'md =' + str(md)
print 'mss = ' + str(mss)
print
