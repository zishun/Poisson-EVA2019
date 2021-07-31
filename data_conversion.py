import numpy as np
from tqdm import tqdm

# setup rpy2 on Windows
# edit the path here according to your machine
import platform
if platform.system() == 'Windows':
    import os
    os.environ['PATH'] = 'C:/Program Files/R/R-3.6.0/bin/' + os.pathsep + 'C:/Program Files/R/R-3.6.0/bin/x64/' + os.pathsep + os.environ['PATH']


import rpy2.robjects as robjects


def RData2npy(rdata_fn, keys, folder_output):
    robjects.r['load'](rdata_fn)
    for key in keys:
        np.save(folder_output+key+'.npy', robjects.r[key])


def convert_rdata():
    print('Convert original rdata to npy files...')

    rdata_fn = './data/DATA_TRAINING.RData'
    keys = ['anom.training', 'loc', 'year', 'month', 'day', 'index.validation']  # 'index.training' is not necessary
    RData2npy(rdata_fn, keys, './data/')

    rdata_fn = './data/TRUE_DATA_RANKING.RData'
    keys = ['X.min.true']
    RData2npy(rdata_fn, keys, './data/')


def export_yyyymmdd():
    print('Process date...')
    # The days have been sorted.
    # Each year has exactly 365 days, no leap year.
    year = np.load('./data/year.npy')
    month = np.load('./data/month.npy')
    day = np.load('./data/day.npy')
    num_days = year.shape[0]
    yyyymmdd = np.zeros((num_days, 3), dtype=np.int32)
    yyyymmdd[:, 0] = year
    yyyymmdd[:, 1] = month
    yyyymmdd[:, 2] = day
    np.save('./data/yyyymmdd.npy', yyyymmdd)


def export_location():
    print('Process location...')
    loc = np.load('./data/loc.npy')
    x = np.unique(loc[:, 0])
    y = np.unique(loc[:, 1])

    loc_int = np.zeros_like(loc, dtype=np.int32)
    x_int = np.zeros(loc.shape[0], dtype=np.int32)
    y_int = np.zeros(loc.shape[0], dtype=np.int32)
    for i in range(x.shape[0]):
        loc_int[np.argwhere(loc[:, 0] == x[i]), 0] = i
    for i in range(y.shape[0]):
        loc_int[np.argwhere(loc[:, 1] == y[i]), 1] = i

    np.save('./data/ind2sub.npy', loc_int)

    num_row = x.shape[0]
    num_col = y.shape[0]
    int2loc_mat = -np.ones((num_row, num_col), dtype=np.int32)
    for i in range(loc_int.shape[0]):
        int2loc_mat[loc_int[i, 0], loc_int[i, 1]] = i
    np.save('./data/sub2ind.npy', int2loc_mat)


# compute the distance in km from longitude/latitude coordinates
# https://github.com/cran/fields/blob/9ddd6d6d22827db57d1983021d5f85563d1a8112/R/rdist.earth.R
def rdist_earth_batch(x1, x2):
    R = 6378.388

    coslat1 = np.cos((x1[1] * np.pi)/180)
    sinlat1 = np.sin((x1[1] * np.pi)/180)
    coslon1 = np.cos((x1[0] * np.pi)/180)
    sinlon1 = np.sin((x1[0] * np.pi)/180)

    coslat2 = np.cos((x2[:, 1] * np.pi)/180)
    sinlat2 = np.sin((x2[:, 1] * np.pi)/180)
    coslon2 = np.cos((x2[:, 0] * np.pi)/180)
    sinlon2 = np.sin((x2[:, 0] * np.pi)/180)

    A = np.empty((x2.shape[0], 3))
    A[:, 0] = coslat2 * coslon2
    A[:, 1] = coslat2 * sinlon2
    A[:, 2] = sinlat2
    pp = A.dot(np.array([coslat1 * coslon1, coslat1 * sinlon1, sinlat1]))
    pp[pp > 1] = 1
    pp[pp < -1] = -1
    return (R * np.arccos(pp))


def find_neighbors():
    loc = np.load('./data/loc.npy')
    radius = 50  # neighborhood radius in kilometers
    # at most max_nei neightbors
    max_nei = 1000
    num_loc = loc.shape[0]
    nei = -np.ones((num_loc, max_nei), dtype=np.int32)
    for i in tqdm(range(num_loc)):
        num_nei = 0
        dist = rdist_earth_batch(loc[i, :], loc)
        idx = np.where(dist < radius)[0]
        nei[i, :idx.shape[0]] = idx
        if idx.shape[0] > max_nei:
            print('wrong!')
    np.save('./data/neighbor.npy', nei)


def onebased2zerobased():
    # 'index.training' is not necessary
    #keys = ['index.training', 'index.validation']
    keys = ['index.validation']
    for key in keys:
        a = np.load('./data/'+key+'.npy')
        np.save('./data/'+key+'0.npy', a-1)


def export_true_observations():
    X_min_true = np.load('./data/X.min.true.npy')
    index_validation = np.load('./data/index.validation0.npy')
    X_min_true = np.reshape(X_min_true, (-1, 1), order='F')
    true_observations = X_min_true[index_validation].reshape(-1)
    np.save('./data/true.observations.npy', true_observations)


if __name__ == '__main__':
    convert_rdata()
    export_yyyymmdd()
    export_location()
    find_neighbors()
    onebased2zerobased()
    export_true_observations()
