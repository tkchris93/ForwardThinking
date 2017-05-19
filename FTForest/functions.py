
import numpy as np
from itertools import product


# windowing
# X shape = (n_examples, dimension of data)
def windowing_tr(X, y, img_size=(28,28), w_size=10):
    n_samples, dim_data = X.shape
    if img_size[0]*img_size[1] != dim_data: # must make sure dim of data is square
        raise ValueError('img_size not compatible with size of data in X')
    n_x = img_size[1]- w_size+1
    n_y = img_size[0]- w_size+1
    num_wind = n_y*n_x
    X_out = np.empty((num_wind*n_samples, w_size**2))
    y_out = np.zeros(num_wind*n_samples)
    for k in xrange(n_samples):
        k_image = X[k,:].reshape(img_size)
        for i in xrange(n_y):
            for j in xrange(n_x):
                i_row = i*n_x
                X_out[k*num_wind + i_row+j,:] = k_image[i:i+w_size, j:j+w_size].flatten()
                y_out[k*num_wind + i_row+j] = y[k]
    return X_out.astype('float32'), y_out.astype('float32')


def windowing_push(X, img_size=(28,28), w_size=10):
    n_samples, dim_data = X.shape
    if img_size[0]*img_size[1] != dim_data: # must make sure dim of data is square
        raise ValueError('img_size not compatible with size of data in X')
    n_x = img_size[1]- w_size+1
    n_y = img_size[0]- w_size+1
    num_wind = n_y*n_x
    X_out = np.empty((num_wind*n_samples, w_size**2))
    for k in xrange(n_samples):
        k_image = X[k,:].reshape(img_size)
        for i in xrange(n_y):
            for j in xrange(n_x):
                i_row = i*n_x
                X_out[k*num_wind + i_row+j,:] = k_image[i:i+w_size, j:j+w_size].flatten()
    return X_out.astype('float32')

def push_thru_MGS_windowing(X_data, MGS_forests, img_size=(28,28), windows=[7,9,14]):
    n_classes = MGS_forests[0].n_classes_
    n_samples = X_data.shape[0]
    n_x = img_size[1]+1 -np.array(windows)
    n_y = img_size[0]+1 -np.array(windows)
    N = n_x * n_y
    X7 = windowing_push(X_data, w_size=7)
    pred_all1 = MGS_forests[0].predict_proba(X7)
    pred_all1 = pred_all1.reshape(n_samples, N[0]*n_classes)
    pred_all2 = MGS_forests[1].predict_proba(X7)
    X7 = 0
    pred_all2 = pred_all2.reshape(n_samples, N[0]*n_classes)
    pred_all1 = np.hstack((pred_all1, pred_all2))
    X9 = windowing_push(X_data, w_size=9)
    pred_all2 = MGS_forests[2].predict_proba(X9)
    pred_all2 = pred_all2.reshape(n_samples, N[1]*n_classes)
    pred_all1 = np.hstack((pred_all1, pred_all2))
    pred_all2 = MGS_forests[3].predict_proba(X9)
    X9 = 0
    pred_all2 = pred_all2.reshape(n_samples, N[1]*n_classes)
    pred_all1 = np.hstack((pred_all1, pred_all2))
    X14 = windowing_push(X_data, w_size=14)
    pred_all2 = MGS_forests[4].predict_proba(X14)
    pred_all2 = pred_all2.reshape(n_samples, N[2]*n_classes)
    pred_all1 = np.hstack((pred_all1, pred_all2))
    pred_all2 = MGS_forests[5].predict_proba(X14)
    X14 = 0
    pred_all2 = pred_all2.reshape(n_samples, N[2]*n_classes)
    return  np.hstack((pred_all1, pred_all2)).astype('float32')

def push_thru_MGS_windowing_sep(X_data, MGS_forests, w_size, img_size=(28,28)):
    n_classes = MGS_forests[0].n_classes_
    n_samples = X_data.shape[0]
    n_x = img_size[1]+1 - w_size
    n_y = img_size[0]+1 - w_size
    N = n_x * n_y
    Xw = windowing_push(X_data, w_size=w_size)
    pred_all1 = MGS_forests[0].predict_proba(Xw)
    pred_all1 = pred_all1.reshape(n_samples, N*n_classes)
    pred_all2 = MGS_forests[1].predict_proba(Xw)
    Xw = 0
    pred_all2 = pred_all2.reshape(n_samples, N*n_classes)
    return  np.hstack((pred_all1, pred_all2)).astype('float32')


def push_thru_MGS_windowing_sep_npy(filename, MGS_forests, w_size, write_to_file=False, out_file=None):
    X = np.load(filename)
    n_c = MGSforest1.n_classes_
    n_samples = X.shape[0]
    N = (28-w_size+1)**2
    if write_to_file and out_file == None:
        raise ValueError('if want to write to file, must provide filename')

    # push data thru MGSforests
    pred1 = MGS_forests[0].predict_proba(X)
    pred1 = pred1.reshape(n_samples, N*n_c)
    pred2 = MGS_forests[1].predict_proba(X)
    pred2 = pred1.reshape(n_samples, N*n_c)
    X_thru = np.hstack((pred1, pred2))
    if write_to_file:
        np.save(out_file+'/X_thru_'+str(w_size), X_thru)
        return X_thru
    else:
        return X_thru

def combine_MGS_output(filenames, w_sizes=[7,9,14]):  #filenames is list of filenames corresponding to the data pushed thru MGS_forests
    if len(filenames) != len(w_sizes):
        raise ValueError('Need to have the same number of files as number of windows')
    count = 0
    for fname in filenames: #add in checks that in correct order, 7, 9, 14
        if count == 0:
            X_out = np.load(fname)
            print X_out.shape
        else:
            X = np.load(fname)
            print X.shape
            X_out = np.hstack((X_out, X))
            X = 0 # free up memory
        count += 1
    return X_out.astype('float32')


# single pixel, wiggle
def single_pixeldiag_tr(X, y, img_size=(28,28)):
    n_samples, dim_data = X.shape
    if img_size[0]*img_size[1] != dim_data: # must make sure dim of data is square
        raise ValueError('img_size not compatible with size of data in X')
    X_out = np.empty((n_samples*5, dim_data))
    y_out = np.zeros(n_samples*5)
    for k in xrange(n_samples):
        k_image = X[k,:].reshape(img_size)
        count = 0
        for i,j in list(product([-1,1], [-1,1])) + [(0,0)]:
            image = np.zeros((img_size[0]+2, img_size[1]+2))
            image[1+i:1+i+img_size[0], 1+j:1+j+img_size[1]] = k_image
            X_out[5*k+count] = image[1:1+img_size[0],1:1+img_size[1]].flatten()
            y_out[5*k+count] = y[k]
            count += 1
    return X_out, y_out
