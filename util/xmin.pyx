# Computation of X(s,t)=min A(s,t)
# i.e. Eq.1 in Huser2020Editorial

import numpy as np
from cython.parallel import prange
from libc.math cimport isnan

DTYPE = np.double

cimport cython
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.

# X=minA on the whole spatio-temporal dataset
def X_min_A_compute(double [:,:] data, int [:,:] neighbor):
    X_min_np = np.zeros_like(data) #this matrix will contain the spatio-temporal minimum X(s,t)=min A(s,t); object size: 1441.9 Mb
    cdef double [:,:] X_min = X_min_np
    cdef int num_loc, num_day, num_nei, i, j, day, nei_id, nei_loc
    cdef double min_value, v
    num_loc = data.shape[1]
    num_day = data.shape[0]
    num_nei = neighbor.shape[1]

    for j in range(num_loc): # for each location
        print('%d/%d\r'%(j,num_loc),end='')
        #for i in range(num_day): # for each day
        for i in prange(num_day,nogil=True,schedule='guided'):
            min_value = 10.0
            for day in range(max(0,i-3),min(num_day,i+4)):
                for nei_id in range(num_nei):
                    nei_loc = neighbor[j,nei_id]
                    if nei_loc < 0:
                        break
                    v = data[day,nei_loc]
                    if isnan(v):
                        min_value = v
                        break
                    if v < min_value:
                        min_value = v
            X_min[i,j] = min_value

    print()
    return X_min_np
    
# X=minA on specific locations (eval_loc) given one week data
def X_min_week(double [:,:] data, int [:,:] neighbor, int [:] eval_loc):
    cdef int num_nei, j, day, nei_id, nei_loc
    cdef double min_value, v
    cdef int num_eval_loc = eval_loc.shape[0]
    X_min_np = np.zeros((num_eval_loc,))
    cdef double [:] X_min = X_min_np
    num_nei = neighbor.shape[1]

    #for j in prange(num_eval_loc,nogil=True,schedule='guided'): # for each location
    for j in range(num_eval_loc): # for each location
        #print('%d/%d\r'%(j,num_eval_loc),end='')
        min_value = 10.0
        for day in range(7):
            for nei_id in range(num_nei):
                nei_loc = neighbor[eval_loc[j],nei_id]
                if nei_loc < 0:
                    break
                v = data[day,nei_loc]
                if isnan(v):
                    min_value = v
                    break
                if v < min_value:
                    min_value = v
        X_min[j] = min_value

    #print()
    return X_min_np
    
