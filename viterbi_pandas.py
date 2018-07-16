"""Based off of https://en.wikipedia.org/wiki/Viterbi_algorithm"""
from pandas import DataFrame, Series
from seaborn import light_palette

## Initialize tuples of conditions.  Observations are the input
observations = ("Eating Pizza", "Browsing Reddit", "Drinking Mountain Dew",
                "Eating Doritos", "Wearing Trenchcoat & Fedora")
hidden_states = ("Depressed", "Confident", "Tired", "Hungry",
                 "Thirsty")  # The confounding factors

trans_prob_df = DataFrame(  # Probability of transition
    data={ # From depressed to confident, vice-versa, static, etc.
        "Depressed": (0.20, 0.25, 0.10, 0.20, 0.20),
        "Confident": (0.15, 0.25, 0.10, 0.20, 0.25),
        "Tired": (0.25, 0.10, 0.30, 0.10, 0.15),
        "Hungry": (0.20, 0.20, 0.25, 0.30, 0.30),
        "Thirsty": (0.20, 0.20, 0.25, 0.20, 0.10)
    }, # All should vertically sum to 1
    index=hidden_states)

emit_prob_df = DataFrame(  # Probability of observation given the hidden state
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

### Initialize backtrace dataframe
backtrace_df = DataFrame(
    data={"Previous State 0": (None, None, None, None, None)},
    index=hidden_states)

### Initialize Dynammic Programming Matrix at Probability 0
viterbi_df = start_probs.multiply(emit_prob_df[observations[0]], axis="index")

### Start Dynammic Programming
for i in range(1, len(observations)):  # Offset from Probability 0
    for state in hidden_states:
        # Compute all possible transitions and take the max
        max_trans_prob = max(viterbi_df.iloc[:, i - 1][prev_state] *
                             trans_prob_df.loc[prev_state][state]
                             for prev_state in hidden_states)
        for prev_state in hidden_states:
            # Ensure that the previous state is correctly captured
            if viterbi_df.iloc[:, i -
                               1][prev_state] * trans_prob_df.loc[prev_state][state] == max_trans_prob:
                max_prob = max_trans_prob * emit_prob_df.loc[state][observations[i]]
                # Multiply transition and emission probabilities and update dataframes
                viterbi_df.loc[state, "Probability {}".format(i)] = max_prob
                backtrace_df.loc[state, "Previous State {}".format(
                    i)] = prev_state
                break

### Obtain the highest possible probability and final state
max_resulting_prob = viterbi_df.iloc[:, -1].max()
dyn_prog_path = [viterbi_df.iloc[:, -1].idxmax()]

### Now backtrace
for i in range(len(observations) - 1, 0, -1):  # Countdown from previous
    dyn_prog_path.insert(  # column, then insert max previous state
        0, backtrace_df.iloc[:, i][dyn_prog_path[
            i - (len(observations))]])  # Inverse of countdown
print("The hidden states are most likely " + ' '.join(dyn_prog_path) + \
    " with a final probability of %s" % max_resulting_prob + "\n")

### Print dynammic programming matrix
viterbi_df.style.background_gradient(
    cmap=light_palette("green", as_cmap=True))  # DONE!
