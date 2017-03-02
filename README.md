# my_dl_solutions
My nuts and bolts solutions on some Deep Learning/Machine Learning/NLP projects

## Quora question pair duplication detection
Dataset: 400K+ pairs, task: binary classification [Link to download](https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view?usp=sharing)

**Model Architecture:**
![Model Architecture] (https://raw.githubusercontent.com/lampts/my_dl_solutions/master/QUORA.png)


**Leaderboard**

|Model | # Params | Test (Accuracy)|
|------|----------|----------------|
| Bilateral Multi-Perspective (Wang, IBM)|N/A|88.17|
|Small cat RNN+CNN *this model* (Laam, Sentifi)|1.4M|84.89|

Notes: we need the apple-to-apple comparison, so I just put the result based on same test data from Wang.


## ATIS:  Airline Travel Information System(ATIS) dataset

My solution: F1 on validation is 94.92
