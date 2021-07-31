#####################################################################
# Dan Cheng, Zishun Liu. Spatio-Temporal Prediction of Missing
# Temperature with Stochastic Poisson Equations -- The LC2019 team
# winning entry for the EVA 2019 data competition.
# Extremes 24(1), 2021.
#####################################################################

import sys, os, time, platform
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, eye
from scipy.sparse.linalg import factorized  # turn off scikits.umfpack if you found it slow
from multiprocessing import Pool, sharedctypes
import util  # custom module

NUM_DAYS = 11315
NUM_LOCS = 16703


class FiniteDifferenceMatrixConstruction:
    def __init__(self, ind2sub_fn='./data/ind2sub.npy', sub2ind_fn='./data/sub2ind.npy'):
        # 1-dim index in [0,16702]
        # 2-dim index in [0,232]x[0,358] converted from longitude/latitude
        self.ind2sub = np.load(ind2sub_fn)  # 1-dim index to 2-dim index
        self.sub2ind = np.load(sub2ind_fn)  # 2-dim index to 1-dim index
        self.width, self.height = self.sub2ind.shape  # Caution: transposed!

    def sub2ind_safe(self, p):
        # check if input 2-dim index is in the domain
        if (p[0] >= 0 and p[0] < self.width and
            p[1] >= 0 and p[1] < self.height):
            return self.sub2ind[p[0],p[1]]
        else:
            return -1

    # construct matrices for the Laplace-based solution
    def laplacian_matrix_construction(self, frame, lambda_array=[0]):
        is_unknown = np.isnan(frame)
        unknown = np.cumsum(is_unknown.astype(int))-1
        unknown[~is_unknown] = -1
        num_unknown = np.max(unknown)+1
        lap_i = []; lap_j = []; lap_v = []
        rhs_i = []; rhs_j = []; rhs_v = []
        grad_j = []; grad_rhs_idx = []
        num_neighbor_array = np.zeros((num_unknown,))
        neighbor_offset = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]])
        for i in np.where(is_unknown)[0]:
            neighbors_unknown = []
            neighbors_known = []
            coord = self.ind2sub[i, :]
            for coord_neighbor in coord+neighbor_offset:
                neighbor_idx = self.sub2ind_safe(coord_neighbor)
                if neighbor_idx >= 0:
                    if is_unknown[neighbor_idx]:
                        neighbors_unknown.append(neighbor_idx)
                    else:
                        neighbors_known.append(neighbor_idx)
                        grad_j.append(unknown[i])
                        grad_rhs_idx.append(neighbor_idx)
            num_neighbor = len(neighbors_known) + len(neighbors_unknown)
            num_neighbor_array[unknown[i]] = num_neighbor
            lap_i.extend([unknown[i]]*len(neighbors_unknown))
            lap_j.extend(unknown[np.array(neighbors_unknown, dtype=int)].tolist())
            lap_v.extend([-1]*len(neighbors_unknown))
            rhs_i.extend([unknown[i]]*len(neighbors_known))
            rhs_j.extend(neighbors_known)
            rhs_v.extend([1.0]*len(neighbors_known))

        # diagonal elements
        lap_i.extend(np.arange(num_unknown).tolist())
        lap_j.extend(np.arange(num_unknown).tolist())
        lap_v.extend(num_neighbor_array.tolist())

        lap_i = np.array(lap_i, dtype=int)
        lap_j = np.array(lap_j, dtype=int)
        lap_v = np.array(lap_v)
        rhs_i = np.array(rhs_i, dtype=int)
        rhs_j = np.array(rhs_j, dtype=int)
        rhs_v = np.array(rhs_v)

        num_edge = len(grad_j)
        grad_j = np.array(grad_j, dtype=int)
        grad_i = np.arange(num_edge, dtype=int)
        grad_v = np.ones_like(grad_i)
        grad_rhs_idx = np.array(grad_rhs_idx, dtype=int)

        Lap = csc_matrix((lap_v, (lap_i, lap_j)), shape=(num_unknown, num_unknown))
        Rhs = csr_matrix((rhs_v, (rhs_i, rhs_j)), shape=(num_unknown, frame.shape[0]))
        Grad = csr_matrix((grad_v, (grad_i, grad_j)), shape=(num_edge, num_unknown))

        Lap_Solver_Array = []
        # (Laplace+lambda*Identity) as screened Poisson's eqn
        for lamb in lambda_array:
            # pre-factorize the matrix for faster solving
            Lap_Solver_Array.append(factorized(Lap+lamb*eye(num_unknown)))

        return Lap, Lap_Solver_Array, Rhs, is_unknown, Grad, grad_rhs_idx

    # construct matrices for the least squares
    def gradient_matrix_construction(self, frame):
        is_unknown = np.isnan(frame)
        unknown = np.cumsum(is_unknown.astype(int))-1
        unknown[~is_unknown] = -1
        num_unknown = np.max(unknown)+1
        grad_i = []; grad_j = []; grad_v = []; 
        grad_rhs_idx = []
        cnt_edge = 0
        neighbor_offset = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]])
        for i in np.where(is_unknown)[0]:
            coord = self.ind2sub[i, :]
            for coord_neighbor in coord+neighbor_offset:
                neighbor_idx = self.sub2ind_safe(coord_neighbor)
                if neighbor_idx >= 0:
                    if is_unknown[neighbor_idx]:
                        if unknown[neighbor_idx] > unknown[i]:
                            grad_i.extend([cnt_edge, cnt_edge])
                            grad_j.extend([unknown[i], unknown[neighbor_idx]])
                            grad_v.extend([1.0, -1.0])
                            grad_rhs_idx.append(-1)
                            cnt_edge += 1
                    else:
                        grad_i.append(cnt_edge)
                        grad_j.append(unknown[i])
                        grad_v.append(1.0)
                        grad_rhs_idx.append(neighbor_idx)
                        cnt_edge += 1

        grad_i = np.array(grad_i, dtype=int)
        grad_j = np.array(grad_j, dtype=int)
        grad_v = np.array(grad_v)
        grad_rhs_idx = np.array(grad_rhs_idx, dtype=int)

        G = csr_matrix((grad_v, (grad_i, grad_j)), shape=(cnt_edge, num_unknown))
        # Moore-Penrose pseudo-inverse for least-squares
        # https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Linear_least-squares
        # since GTG==Lap, this part is skipped
        # GTG = G.T.dot(G)
        # G_Solver_Array = []
        # for lamb in lambda_array:
        #     G_Solver_Array.append(factorized(GTG+lamb*eye(num_unknown)))

        return G, grad_rhs_idx

    def matrix_construction(self, frame, lambda_array=[0]):
        Lap, Lap_Solver_Array, Rhs, is_unknown, Grad, grad_rhs_idx = self.laplacian_matrix_construction(frame, lambda_array)
        LS_G, LS_grad_rhs_idx = self.gradient_matrix_construction(frame)

        return Lap, Lap_Solver_Array, Rhs, is_unknown, Grad, grad_rhs_idx, LS_G, LS_grad_rhs_idx


def sample_val_index(year_mask):
    index = np.load('./data/split_all_val_index.npy')
    s = np.floor(index/NUM_DAYS).astype(int)
    t = np.remainder(index, NUM_DAYS).astype(int)
    mask = np.zeros((NUM_DAYS, NUM_LOCS),dtype=bool)
    mask[(t,s)] = True
    for i in range(31):
        if year_mask[i] == False:
            mask[365*i:365*(i+1),:] = False
    val_index = np.where(mask)
    return val_index


def load_official_val_index():
    index = np.load('./data/index.validation0.npy')
    s = np.floor(index/NUM_DAYS).astype(int)
    t = np.remainder(index, NUM_DAYS).astype(int)
    return (t, s)


def gen_val(index):
    X_min_true_flip = np.load('./data/X_min_flip.npy')
    y = X_min_true_flip[index]
    return y


class X_min():

    def __init__(self, neighbor_fn='./data/neighbor.npy'):
        self.neighbors = np.load(neighbor_fn).astype(np.int32)

    def X_min(self, data, eval_loc):
        x_min = util.X_min_week(data, self.neighbors, eval_loc.astype(np.int32))
        return x_min


def use_lap_to_fill(sample_idx):
    # Read-only global variables: 
    # Lap, Lap_Solver_Array, Rhs, is_unknown, Grad, grad_rhs_idx, LS_G, LS_grad_rhs_idx
    # XMIN, true_data, split_train, num_lambda, val_index, val_days_this_month, samples_weeks
    # lap_sol_dict
    # Write to: simulated_x_min_shared

    simulated_x_min = np.ctypeslib.as_array(simulated_x_min_shared)
    first_day = samples_weeks[sample_idx]*7

    # prepare Laplace
    lap_week = Lap.dot(true_data[is_unknown, first_day:first_day+7])
    lap_week_rhs = Rhs.dot(true_data[:, first_day:first_day+7])
    mask = np.logical_or(np.isnan(lap_week), np.isnan(lap_week_rhs))
    lap_week[mask] = 0.0
    lap_week_rhs[mask] = 0.0
    lap_unknown = lap_week - lap_week_rhs
    num_unknown = np.sum(is_unknown)

    # prepare least-square
    ref_grad_rhs = true_data[LS_grad_rhs_idx, first_day:first_day+7]
    ref_grad_rhs[LS_grad_rhs_idx < 0, :] = 0.0
    ref_grad_lhs = LS_G.dot(true_data[is_unknown, first_day:first_day+7])
    mask = np.logical_or(np.isnan(ref_grad_lhs), np.isnan(ref_grad_rhs))
    ref_grad_lhs[mask] = 0.0
    ref_grad_rhs[mask] = 0.0
    ref_grad = ref_grad_lhs - ref_grad_rhs

    # solve Poisson's eqn.
    # Laplacian(u) = f, u = 0 on boundary
    poisson_sol = np.empty((num_unknown, 7, num_lambda), order='F')
    for i in range(num_lambda):
        poisson_sol[:,:,i] = Lap_Solver_Array[i](lap_unknown)

    for day in val_days_this_month:
        val_idx_day = np.where(val_index[0]==day)[0]
        val_loc = val_index[1][val_idx_day]

        lap_sol = lap_sol_dict[day]
        ls_sol = np.empty((num_unknown,7,num_lambda),order='F')
        completion_week = split_train[:,day-3:day+4].copy()
        edge_norms = np.empty((num_lambda,7))
        ls_edge_norms = np.empty((num_lambda,7))
        for i in range(num_lambda):
            edge_norms[i,:] = np.linalg.norm(Grad.dot((lap_sol[:,:,i] + poisson_sol[:,:,i]))-split_train[grad_rhs_idx,day-3:day+4], axis=0)

        # least square solution
        grad_rhs = split_train[LS_grad_rhs_idx,day-3:day+4].copy()
        grad_rhs[LS_grad_rhs_idx<0,:] = 0.0
        ls_solve_rhs = LS_G.T.dot(grad_rhs+ref_grad)
        for i in range(num_lambda):
            ls_sol[:,:,i] = Lap_Solver_Array[i](ls_solve_rhs)
            ls_edge_norms[i,:] = np.linalg.norm(Grad.dot(ls_sol[:,:,i])-split_train[grad_rhs_idx,day-3:day+4], axis=0)
        
        for i in range(7):
            # for each day, select that solution with minimum "boundary step"
            m1 = np.argmin(edge_norms[:,i])
            m2 = np.argmin(ls_edge_norms[:,i])
            if edge_norms[m1,i] < ls_edge_norms[m2,i]:
                completion_week[is_unknown,i] = lap_sol[:,i,m1] + poisson_sol[:,i,m1]
            else:
                completion_week[is_unknown,i] = ls_sol[:,i,m2]
        # end of 7 days in week

        x_min = XMIN.X_min(completion_week.T, val_loc)
        simulated_x_min[val_idx_day, sample_idx] = x_min
    # end of validation days in month
    return

if __name__ == '__main__':
    # Usage:
    #   python3 ./PoissonTemperature.py <year> <#samples>
    #   year = -1 for submission; 0...21 for cross-validation

    VAL_YEAR = int(sys.argv[1])
    NUM_SAMPLE = int(sys.argv[2])
    if VAL_YEAR >= 0:
        SUBMISSION = False
        print('Use year %d for validation'%(int(sys.argv[1])))
        year_mask = np.zeros((31,),dtype=bool)
        year_mask[VAL_YEAR] = True
    else:
        SUBMISSION = True
        print('Submission')

    # difficult to manage shared memory for multiprocessing on Windows
    MP = not (platform.system() == 'Windows' or 'microsoft' in platform.uname()[3].lower())
    XMIN = X_min()
    FDMC = FiniteDifferenceMatrixConstruction()
    util_score = util.UtilScore()

    # load data
    yyyymmdd = np.load('./data/yyyymmdd.npy')
    true_data = np.load('./data/anom.training.npy').T  # each column for a day
    if SUBMISSION:
        split_train = true_data
        val_index = load_official_val_index()  # validation time/space index
    else:
        split_train = -np.load('./data/split_all.npy').T  # each column for a day
        val_index = sample_val_index(year_mask)
        true_observations = gen_val(val_index)
    val_days = np.unique(val_index[0])
    num_val = val_index[0].shape[0]

    # clone gradient/Laplace fields by week from the 22 reference/training years
    max_num_lap_fields = 22*365//7
    np.random.seed(0)
    samples_weeks = np.random.choice(max_num_lap_fields, NUM_SAMPLE, replace=False)
    # store all x_min estimated for each validation point
    simulated_x_min = np.empty((num_val,NUM_SAMPLE))
    simulated_x_min_ctypes = np.ctypeslib.as_ctypes(simulated_x_min)
    simulated_x_min_shared = sharedctypes.RawArray(simulated_x_min_ctypes._type_, simulated_x_min_ctypes)

    # lambdas in screened Poisson's eqn and regularized least-squares.
    lambda_array = 0.02*np.power(0.5,np.arange(13)) # around [2e-2, ..., 1e-5]
    lambda_array[-1] = 0.0

    start_time = time.time()

    cnt_month = 0
    print('[%s] start'%(time.ctime()))
    for month in range(12*31):
        # find all validation points in the same month and solve them together
        # as they share the same linear systems.

        same_year = (yyyymmdd[val_days,0] == 1985+month//12)
        same_month = (yyyymmdd[val_days,1] == 1+month%12)
        val_days_this_month = val_days[np.logical_and(same_year, same_month)]
        if val_days_this_month.size == 0:
            continue # no validation day falls in this month

        same_year = (yyyymmdd[:,0] == 1985+month//12)
        same_month = (yyyymmdd[:,1] == 1+month%12)
        first_day_of_the_month = np.where(np.logical_and(same_year, same_month))[0][0]
        Lap, Lap_Solver_Array, Rhs, is_unknown, Grad, grad_rhs_idx, LS_G, LS_grad_rhs_idx = FDMC.matrix_construction(split_train[:,first_day_of_the_month], lambda_array)

        lap_sol_dict = {}
        num_lambda = lambda_array.size
        num_unknown = np.sum(is_unknown)
        for day in val_days_this_month:
            rhs = Rhs.dot(split_train[:,day-3:day+4])
            # solve Laplace's eqn.
            # Laplacian(u) = 0, u = u* on boundary
            lap_sol = np.empty((num_unknown,7,num_lambda),order='F')
            for i in range(num_lambda):
                lap_sol[:,:,i] = Lap_Solver_Array[i](rhs)
            lap_sol_dict[day] = lap_sol

        if MP:
            with Pool() as p:
                p.map(use_lap_to_fill, range(NUM_SAMPLE)) 
        else:
            for i in range(NUM_SAMPLE):
                use_lap_to_fill(i)
        print('[%s] %d'%(time.ctime(),cnt_month))
        cnt_month += 1

    simulated_x_min = np.ctypeslib.as_array(simulated_x_min_shared)

    prediction = np.zeros((num_val, 400))
    for i in range(num_val):
        prediction[i,:] = util_score.ecdf(simulated_x_min[i,:])

    print('Inference: %.2fs'%(time.time()-start_time))

    output_folder = './data/result/'
    os.makedirs(output_folder, exist_ok=True)
    if SUBMISSION:
        np.save('%s/simulated_x_min_val.npy'%(output_folder), simulated_x_min)
        np.save('%s/final_prediction.npy'%(output_folder), prediction)
        try:
            true_observations = np.load('./data/true.observations.npy')
            score = util_score.twCRPS(prediction,true_observations)
            print('score %.3fe-4'%(score*1.e4))
        except:
            print('Cannot load true observations.')
    else:
        np.save('%s/simulated_x_min_%d.npy'%(output_folder, VAL_YEAR), simulated_x_min)
        score = util_score.twCRPS(prediction,true_observations)
        print('score %.3fe-4'%(score*1.e4))
