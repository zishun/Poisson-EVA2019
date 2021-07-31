import numpy as np
import time
import util

fn_input = './data/anom.training.npy'
fn_output = './data/X_min_flip.npy'
data = -np.load(fn_input)

neighbors = np.load('./data/neighbor.npy').astype(np.int32)
start = time.time()
X_min = util.X_min_A_compute(data, neighbors)
print('compute X_min: %.3fs' % (time.time()-start))
np.save(fn_output, X_min)
