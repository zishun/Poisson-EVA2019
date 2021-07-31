import numpy as np
import imageio
from PoissonTemperature import FiniteDifferenceMatrixConstruction


def ind_sub_conversion(img, ind2sub_fn, sub2ind_fn):
    rows, cols = img.shape[:2]
    num = rows*cols
    arange = np.arange(rows*cols, dtype=np.int32)
    ind2sub = np.empty((num, 2), dtype=np.int32)
    ind2sub[:, 0] = np.floor(arange/cols)
    ind2sub[:, 1] = np.remainder(arange, cols)
    sub2ind = arange.reshape((rows, cols))

    np.save(ind2sub_fn, ind2sub)
    np.save(sub2ind_fn, sub2ind)


def pie(FDMC, background, foreground):
    Lap, Lap_Solver_Array, Rhs, is_unknown, _, _ = \
        FDMC.laplacian_matrix_construction(mask.ravel())
    bg = background.reshape((-1, 3))
    fg = foreground.reshape((-1, 3))
    result = bg.copy()

    lap = Lap.dot(fg[is_unknown, :])
    lap_rhs = Rhs.dot(fg)
    lap_unknown = lap - lap_rhs
    poisson_sol = Lap_Solver_Array[0](lap_unknown+Rhs.dot(bg))
    result[is_unknown, :] = poisson_sol
    result = result.reshape(background.shape)
    result[result < 0] = 0.0
    result[result > 1] = 1.0
    return (result*255).astype(np.uint8)


if __name__ == '__main__':
    folder = './data/pie/'
    mask = imageio.imread(folder+'mask.png')[:, :, 0].astype(np.float32)
    background = imageio.imread(folder+'mona.png')[:, :, :3]/255
    foreground = imageio.imread(folder+'gine.png')[:, :, :3]/255
    mask[mask > 0] = np.nan

    ind2sub_fn = folder+'ind2sub.npy'
    sub2ind_fn = folder+'sub2ind.npy'
    ind_sub_conversion(mask, ind2sub_fn, sub2ind_fn)

    FDMC = FiniteDifferenceMatrixConstruction(ind2sub_fn, sub2ind_fn)
    result = pie(FDMC, background, foreground)
    imageio.imwrite(folder+'result.png', result)
