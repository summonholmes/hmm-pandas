# hmm-pandas
![alt text](https://raw.githubusercontent.com/summonholmes/hmm-pandas/master/example.png)
Regarding the Hidden Markov Model (HMM) and its associated algorithms (Forward, Forward-Backward, Viterbi), much of the research, materials, and implementations are mathematically convoluted.  This project presents the HMM and its capabilities using Pandas dataframes.  The intention is to simplify the HMM as much as possible, and to create a visually appealing representations of the HMM; while preserving the functionality of existing implementations.  In addition, detailed comments are provided throughout the code.  Therefore, the user should have much less trouble understanding the basic principles behind HMMs.

For the Viterbi algorithm, the observations describe your average programmer.  The hidden states describe how the programmer is feeling.  This script will predict the most likely sequence of hidden states for the programmer.

This project is a work in progress, and only the Viterbi algorithm has been completed.  Later on, this project will also attempt to translate the algorithms into their vectorized counterparts.

## Getting Started
This program requires few dependences and should be trivial to set up.  However, an in-depth understanding of the HMM and its associated algorithms requires some knowledge of bioinformatics, data science, and machine learning.

## Notes
Occassionally, there may be ties that occur during the traceback phase of the dynamic programming process.  This program lets Pandas stochastically select the tie breaker.

### Dependencies
1. Install the following dependencies:
* python3-pandas
* python3-seaborn

### Usage:
#### Mac/Linux/Unix
```
$ python3 /path/to/viterbi_pandas.py
```
#### Windows
```
C:\path\to\main.py C:\path\to\python3.exe viterbi_pandas.py
```
