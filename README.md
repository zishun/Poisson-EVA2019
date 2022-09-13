# Spatio-temporal prediction of missing temperature with stochastic Poisson equations

This is the implementation of the LC2019 team winning entry for the [EVA 2019 data competition](https://web.math.pmf.unizg.hr/eva2019/competitions-1/data-challenge), as described in the [paper](https://doi.org/10.1007/s10687-020-00397-w).


## Requirements
This project was implemented on Linux, on which it should work smoothly.
It also works on Windows, without multiprocessing support (neither on WSL).

This project is mainly implemented with Python3. R is used to convert data. Programming environments of Python3 and R should be setup first. The Python dependencies are listed in ```requirements.txt```. On a Ubuntu machine, R can be installed with
```
sudo apt install r-base
```

## Usage
* Data preparation.
Two data files can be downloaded via the [link](https://github.com/BlackBox-EVA2019/BlackBox/blob/83d63fc3880d0835a776b2ddbf2f4a2369d17957/DATA_FILES.txt).
    * ```DATA_TRAINING.RData```: for training and inference
    * ```TRUE_DATA_RANKING.RData```: only for final evaluation. 

    Put them in ```./data/``` folder.
Convert them to ```.npy``` and prepare all necessary data.
    ```
    python data_conversion.py
    ```

* Build the utilities
    ```
    cd util
    python xmin_setup.py build_ext --inplace
    ```
* Compute ```x_min``` and split cross validation sets
    ```
    python compute_X_min.py
    python split_cross_validation.py
    ```
* Inference
    ```
    python PoissonTemperature.py 0 1000  # score: 3.61e-4
    ```

## Bonus
The code here implements image cloning proposed in [Poisson image editing](https://doi.org/10.1145/882262.882269). Try it with:
```
python poisson_image_editing.py
```
or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zishun/Poisson-EVA2019/blob/main/Poisson_image_editing.ipynb).


<img src="https://github.com/zishun/Poisson-EVA2019/raw/main/data/pie/pie_result.png" width="800"/>

## Bibtex
```
@article{PoissonEVA2019,
    title = {Spatio-temporal prediction of missing temperature with stochastic {P}oisson equations},
    author = {Cheng, Dan and Liu, Zishun},
    year = 2021,
    journal = {Extremes},
    pages = {163--175},
    volume = {24},
    number = {1},
    issn = {1572-915X},
    url = {https://doi.org/10.1007/s10687-020-00397-w},
    doi = {10.1007/s10687-020-00397-w}
}
```

## Related Projects
* [BeatTheHeat](https://github.com/dcastrocamilo/EVAChallenge2019)
* [BlackBox](https://github.com/BlackBox-EVA2019/BlackBox)
* [Multiscale](https://github.com/Joonpyo-Kim/QFM)
