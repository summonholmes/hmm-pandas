from pandas import DataFrame, IndexSlice
from seaborn import light_palette

### Initialize tuples of conditions.  Observations are the input
observations = (  # Modify, add, remove with any key in emit_prob_df
    "Wearing Trenchcoat & Fedora", "Browsing Reddit", "Drinking Mountain Dew",
    "Eating Doritos", "Eating Pizza")
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
    data={"(0) {}".format(observations[0]): (0.10, 0.40, 0.10, 0.20, 0.20)},
    index=hidden_states)

### Initialize dynammic programming matrix at probability 0
viterbi_df = start_probs.multiply(emit_prob_df[observations[0]], axis="index")

### Start dynammic programming
for i, observation in enumerate(observations[1:]):
    max_trans_prob_df = trans_prob_df.multiply(  # Offset by 1
        viterbi_df.iloc[:, i], axis="index").max()
    # Multiply entire trans_prob df by previous viterbi_df
    # column and take vertical maximums
    viterbi_df["({}) {}".format(
        i + 1,  # Then multiply the result by the observation emissions
        observation)] = max_trans_prob_df * emit_prob_df.loc[:, observation]

### Provide the entire matrix with highest values darkest
viterbi_traceback_df = viterbi_df.style.background_gradient(
    cmap=light_palette("green", as_cmap=True))

### At the last column, use the maximum value to begin traceback
traceback_prob = [viterbi_df.iloc[:, -1].max()]
dyn_prog_path = [viterbi_df.iloc[:, -1].idxmax()]  # And its index
viterbi_traceback_df.highlight_max(  # Highlight it
    color="red", subset=IndexSlice[[viterbi_df.columns[-1]]])

### Start traceback
for i, observation in zip(  # Reverse enumerate with offset
        range(len(observations) - 2, -1, -1), reversed(observations[1:])):
    # Isolate the previous location that gives the current probability
    traceback_loc = viterbi_df.loc[ # Always going left-most
        viterbi_df.iloc[:, i] * trans_prob_df.loc[:, dyn_prog_path[0]] *
        emit_prob_df.loc[dyn_prog_path[0], observation] == traceback_prob[
            0]].index[0]
    # Record the value and its state
    traceback_prob.insert(0,
                          viterbi_df.loc[traceback_loc, viterbi_df.columns[i]])
    dyn_prog_path.insert(0, traceback_loc)
    viterbi_traceback_df = viterbi_traceback_df.applymap(
        lambda x: "background-color: red",  # Color the path red
        subset=IndexSlice[[dyn_prog_path[0]], [viterbi_df.columns[i]]])

### Print dynammic programming matrix and traceback results
print("The observations:", ", ".join(observations))
print("The most likely sequence of hidden states is:")
print((viterbi_df.isin(traceback_prob) == True).idxmax())
print("The final probability:", traceback_prob[-1])
viterbi_traceback_df
