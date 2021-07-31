import numpy as np

NUM_DAYS = 11315


def gen_mask_each_month():
    yyyymmdd = np.load('./data/yyyymmdd.npy')
    data = np.load('./data/anom.training.npy')
    first_days = np.where(yyyymmdd[:,2]==1)[0]
    np.save('./data/mask_each_month.npy', np.isnan(data[first_days,:]))

    index = np.load('./data/index.validation.npy')
    s = np.floor(index/NUM_DAYS).astype(int)
    t = np.remainder(index, NUM_DAYS).astype(int)
    validation_each_month = -np.ones((31*12,3,500),dtype=int)

    days = np.unique(t)
    for day in days:
        locs = s[np.where(t==day)[0]]
        month_id = yyyymmdd[day,1] - 1 + 12*(yyyymmdd[day,0]-yyyymmdd[0,0])
        day_id = yyyymmdd[day,2]//10
        validation_each_month[month_id,day_id,:] = locs

    np.save('./data/validation_each_month.npy', validation_each_month)


def nearest(p, mat):
    diff = np.sum(np.logical_xor(mat, p[np.newaxis,:]), axis=1)
    min_id = np.argmin(diff)
    return min_id


def split_all():
    large_masks = mask_each_month[22*12:,:]

    split_val_index_s = []
    split_val_index_t = []

    split_train = data.copy()
    # 1985 - 2015, 2007 is 22
    num_years = 31
    for year in range(num_years):
        print('%d/%d\r' % (year, num_years), end='')
        for month in range(12):
            month_id = year*12+month
            mask_this_month = mask_each_month[month_id, :]
            nearest_large_mask_id = nearest(mask_this_month, large_masks)
            nearest_large_mask = large_masks[nearest_large_mask_id, :]
            date_mask_this_month = np.logical_and(yyyymmdd[:,0]==year+1985, yyyymmdd[:,1]==month+1)
            date_this_month = np.where(date_mask_this_month)[0]
            for day in date_this_month:
                split_train[day,nearest_large_mask] = np.nan
            for day_id in range(3):
                validation_loc = validation_each_month[nearest_large_mask_id+22*12,day_id,:]
                # find the dates: 5, 15, and 25
                day = np.where(np.logical_and(date_mask_this_month, yyyymmdd[:,2]==day_id*10+5))[0][0]
                # find valid locations
                valid_locs = np.logical_not(np.isnan(X_min[day,:]))
                validation_loc_mask = np.zeros_like(valid_locs,dtype=bool)
                validation_loc_mask[validation_loc] = True
                validation_loc_mask = np.logical_and(validation_loc_mask,valid_locs)
                validation_loc_list = (np.where(validation_loc_mask)[0]).tolist()
                num = len(validation_loc_list)
                validation_t_list = [day]*num

                split_val_index_s.extend(validation_loc_list)
                split_val_index_t.extend(validation_t_list)
    print()

    split_val_X_min = X_min[(split_val_index_t, split_val_index_s)]
    split_val_index_s = np.array(split_val_index_s, dtype=int)
    split_val_index_t = np.array(split_val_index_t, dtype=int)
    split_val_index = split_val_index_s*NUM_DAYS+split_val_index_t

    np.save('./data/split_all.npy', split_train)
    np.save('./data/split_all_val_index.npy', split_val_index)
    np.save('./data/split_all_val_observation.npy', split_val_X_min)

if __name__ == '__main__':

    data = np.load('./data/anom.training.npy')
    X_min = np.load('./data/X_min_flip.npy')
    yyyymmdd = np.load('./data/yyyymmdd.npy')
    gen_mask_each_month()  # comment out this line if the following two files have been generated
    mask_each_month = np.load('./data/mask_each_month.npy')
    validation_each_month = np.load('./data/validation_each_month.npy')  # (31*12,3,500)

    split_all()
