"""Based off of https://en.wikipedia.org/wiki/Viterbi_algorithm"""
from pandas import DataFrame, Series
from seaborn import light_palette

## Initialize tuples of conditions, observations is the input
observations = ("Eating Pizza", "Browsing Reddit", "Drinking Mountain Dew",
                "Eating Doritos", "Wearing Trenchcoat & Fedora")
hidden_states = ("Depressed", "Confident", "Tired", "Hungry",
                 "Thirsty")  # The confounding factors

trans_prob_df = DataFrame(  # Probability of transition
    data={ # from health to fever, vice-versa, or static
        "Depressed": (0.20, 0.25, 0.1, 0.2, 0.2),
        "Confident": (0.15, 0.25, 0.1, 0.2, 0.25),
        "Tired": (0.25, 0.1, 0.3, 0.1, 0.15),
        "Hungry": (0.2, 0.2, 0.25, 0.3, 0.3),
        "Thirsty": (0.2, 0.2, 0.25, 0.2, 0.1)
    }, # All should vertically sum to 1
    index=hidden_states)

emit_prob_df = DataFrame(  # Probability of event
    data={ # Normal when healthy, normal when 
        "Eating Pizza": (0.2, 0.1, 0.1, 0.35, 0.2), # fever, etc.
        "Browsing Reddit": (0.2, 0.1, 0.35, 0.1, 0.2),
        "Drinking Mountain Dew": (0.3, 0.1, 0.3, 0.2, 0.3),
        "Eating Doritos": (0.2, 0.1, 0.15, 0.15, 0.15),
        "Wearing Trenchcoat & Fedora": (0.1, 0.6, 0.1, 0.2, 0.15),
    }, # All should vertically sum to 1
    index=hidden_states)

### Initialize starting probabilities
start_probs = DataFrame(
    data={"Probability 0": (0.1, 0.4, 0.1, 0.2, 0.2)}, index=hidden_states)

### Initialize Dynammic Programming Matrix at Probability 0
viterbi_df = start_probs.multiply(emit_prob_df[observations[0]], axis="index")
# viterbi_df = viterbi_df.assign(**{"Previous State 0": None})

## Begin Dynamic Programming
for i in range(1, len(observations)):
    for state in hidden_states:
        max_trans_prob = max(viterbi_df.iloc[:, i - 1][prev_state] *
                             trans_prob_df.loc[prev_state][state]
                             for prev_state in hidden_states)
        for prev_state in hidden_states:
            max_prob = max_trans_prob * emit_prob_df.loc[state][observations[i]]
            viterbi_df.loc[state, "Probability {}".format(i)] = max_prob

### Obtain the highest possible probability and final state
max_resulting_prob = viterbi_df.iloc[:, -1].max()
dyn_prog_path = [viterbi_df.iloc[:, -1].idxmax()]

### Now backtrace
for i in range(len(observations) - 2, -1, -1):  # Countdown from previous
    dyn_prog_path.insert(  # column, then insert max previous state
        0, viterbi_df.iloc[:, i].idxmax())  # at index 0
print("The hidden states are most likely " + ' '.join(dyn_prog_path) + \
    " with a final probability of %s" % max_resulting_prob + "\n") # Currently wrong

### Print dynammic programming matrix
viterbi_df.style.background_gradient(
    cmap=light_palette("green", as_cmap=True))  # DONE!
