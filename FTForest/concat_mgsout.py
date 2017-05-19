import numpy as np
from functions import combine_MGS_output

# read in the pushed thru training data from all window sizes
filenames = ['./7_wind/X_tr_thru_7_wind.npy', './9_wind/X_tr_thru_9_wind.npy', './14_wind/X_tr_thru_14_wind.npy']


X_out = combine_MGS_output(filenames)
np.save('X_mgsout.npy', X_out)
X_out = 0 # Clear Up Memory

filenames = ['./7_wind/X_t_thru_7_wind.npy', './9_wind/X_t_thru_9_wind.npy', './14_wind/X_t_thru_14_wind.npy']
X_t_out = combine_MGS_output(filenames)
np.save('X_t_mgsout.npy', X_t_out)
X_t_out = 0 # Clear Up Memory
