# hmm-pandas
Regarding the Hidden Markov Model (HMM) and its algorithms, much of the research, materials, and implementations are mathematically convoluted.  This project presents the HMM and its algorithms using Pandas dataframes.  The objectives of this project are to demonstrate the HMM using a simple, sane Python implementation; to make the HMM understandable to anyone; to create visually appealing representations; and to improve the performance of existing implementations.  In addition, detailed comments are provided throughout the code.

## Viterbi
![alt text](https://raw.githubusercontent.com/summonholmes/hmm-pandas/master/viterbi.png)

## Forward-Backward
![alt text](https://raw.githubusercontent.com/summonholmes/hmm-pandas/master/forward_backward.png)

For the Viterbi and Forward-Backward algorithms, a set of observations and hidden states are defined.  The observations describe your average programmer and his outward appearance.  What is the programmer wearing, eating, or drinking?  The hidden states describe how the programmer is feeling.  Does the programmer have a high and mighty attitude?  When provided a sequence of observations, the 'viterbi_pandas.py' script will predict the most likely sequence of hidden states for the programmer, while the 'forward_backward_pandas.py' script will calculate the posterior marginals for all hidden states.  Need more verbosity?  Use the 'adv' versions!

Unlike previous implementations, this project utilizes a vectorized approach towards dynamic programming.  Therefore, the scripts run much faster and more efficiently than implementations that loop continuously.  The only source of iteration is the sequence of observations.

## Getting Started
This project requires few dependences and should be trivial to set up.  However, an in-depth understanding of the HMM and its associated algorithms requires some knowledge of probability theory, data science, and dynamic programming.

## Notes
Occassionally, there may be ties that occur during the dynamic programming process.  Pandas selects the upper-most value in the column to determine the tie breaking hidden state.

### Dependencies
* python3-pandas
* python3-seaborn

### Usage:
I'd recommend using these scripts interactively with Jupyter Notebook via VSCode, Atom's Hydrogen, Sublime's Hermes, Pycharm, or your web browser.  Spyder and/or IPython will also work.  Do not use standard Python.
