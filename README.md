# my_dl_solutions
My nuts and bolts solutions on some Deep Learning/Machine Learning/NLP projects

## Quora question pair duplication detection
Dataset: 400K+ pairs, task: binary classification

![Learning curve](https://raw.githubusercontent.com/lampts/my_dl_solutions/master/quora_wang_dataset.png)



Description:

- dataset: https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view?usp=sharing
- Wang achieved accuray 88.17 on his partitioning dataset above


Baseline Solution: CNN + RNN + MLP, 290K parameters.
Baseline metrics: (score + std)

On Wang's dataset: (params 690K)

- loss      = 0.3581
- accuracy  = 0.8383
- precision = 0.8104
- recall    = 0.8940
- F         = 0.8474

1.6M params:

- loss      = 0.3599
- accuracy  = 0.8489
- precision = 0.8471
- recall    = 0.8612
- F         = 0.8505

On Pallen's split: https://github.com/bradleypallen

- params (600K)
- loss      = 0.3608
- accuracy  = 0.8336
- precision = 0.7516
- recall    = 0.8228
- F         = 0.7782

On my own split:

- Precision:  0.786957808057 0.00194049959694
- Recall:  0.786781782452 0.00179403404887
- F1:  0.786868713966 0.00162766538818
- AUC:  0.92140834892 0.000922507254995


## ATIS:  Airline Travel Information System(ATIS) dataset

My solution: F1 on validation is 94.92
