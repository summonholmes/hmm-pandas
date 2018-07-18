"""Based off of https://en.wikipedia.org/wiki/Viterbi_algorithm"""
from pandas import DataFrame, IndexSlice
from numpy import where, max as npmax
from seaborn import light_palette

### Initialize tuples of conditions.  Observations are the input
observations = ("Eating Pizza", "Browsing Reddit", "Drinking Mountain Dew",
                "Eating Doritos", "Wearing Trenchcoat & Fedora")
hidden_states = ("Depressed", "Confident", "Tired", "Hungry",
                 "Thirsty")  # The confounding factors

### Probability of transition
trans_prob_df = DataFrame(
    data={  # From depressed to confident, vice-versa, static, etc.
        "Depressed": (0.20, 0.25, 0.10, 0.20, 0.20),
        "Confident": (0.15, 0.25, 0.10, 0.20, 0.25),
        "Tired": (0.25, 0.10, 0.30, 0.10, 0.15),
        "Hungry": (0.20, 0.20, 0.25, 0.30, 0.30),
        "Thirsty": (0.20, 0.20, 0.25, 0.20, 0.10)
    },  # All should vertically sum to 1
    index=hidden_states)

### Probability of observation given the hidden state
emit_prob_df = DataFrame(
    data={ # Highest chance of trenchcoat & fedora is when confident
        "Eating Pizza": (0.20, 0.10, 0.10, 0.35, 0.20),
        "Browsing Reddit": (0.20, 0.10, 0.35, 0.10, 0.20),
        "Drinking Mountain Dew": (0.30, 0.10, 0.30, 0.20, 0.30),
        "Eating Doritos": (0.20, 0.10, 0.15, 0.15, 0.15),
        "Wearing Trenchcoat & Fedora": (0.10, 0.60, 0.10, 0.20, 0.15),
    }, # All should vertically sum to 1
    index=hidden_states)

### Initialize starting probabilities
start_probs = DataFrame(
    data={"Probability 0": (0.10, 0.40, 0.10, 0.20, 0.20)},
    index=hidden_states)

### Initialize dynammic programming matrix at probability 0
viterbi_df = start_probs.multiply(emit_prob_df[observations[0]], axis="index")

### Start dynammic programming
for i in range(1, len(observations)):  # Offset from Probability 0
    max_trans_prob_df = trans_prob_df.multiply(
        viterbi_df.iloc[:, i - 1], axis="index").apply(
            npmax, axis=0)  # Vectorize maximums for the previous column
    viterbi_df["Probability {}".format(  # Each column all in one go
        i)] = max_trans_prob_df * emit_prob_df.loc[:, observations[i]]

### At the last column, use the maximum value to begin traceback
traceback_prob = [viterbi_df.iloc[:, -1].max()]
dyn_prog_path = [viterbi_df.iloc[:, -1].idxmax()]  # And its index

### Start traceback
for i in range(len(observations) - 1, 0, -1):  # Countdown
    # Isolate the previous location that gives the current probability
    traceback_loc = where(viterbi_df.iloc[:, i - 1] *
                          trans_prob_df.loc[:, dyn_prog_path[0]] *
                          emit_prob_df.loc[dyn_prog_path[0], observations[i]]
                          == traceback_prob[0])[0][0]
    # Record the value and its state
    traceback_prob.insert(0, viterbi_df.iloc[traceback_loc, i - 1])
    dyn_prog_path.insert(0, viterbi_df.index[traceback_loc])

### Provide the entire matrix with highest values darkest
viterbi_traceback_df = viterbi_df.style.background_gradient(
    cmap=light_palette("green", as_cmap=True))

### Now index using the generated lists and columns
for i in range(len(observations)):
    viterbi_traceback_df = viterbi_traceback_df.applymap(
        lambda x: "background-color: red",  # Color the path red
        subset=IndexSlice[[dyn_prog_path[i]], [viterbi_df.columns[i]]])

### Print dynammic programming matrix and traceback results
print("The observations:", ', '.join(observations))
print("The hidden states are most likely: " + ', '.join(dyn_prog_path) + \
    "; with a final probability of %s" % traceback_prob[-1] + "\n")
viterbi_traceback_df
