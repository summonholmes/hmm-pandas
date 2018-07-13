"""Example taken from https://en.wikipedia.org/wiki/Viterbi_algorithm"""
from pandas import DataFrame

### Initialize tuples of conditions, observations is the input
observations = ('Normal', 'Cold', 'Dizzy')
hidden_states = ("Healthy", "Fever")  # The confounding factors

### Initialize all dataframes
trans_prob_df = DataFrame(  # Probability of transition
    data={ # from health to fever, vice-versa, or static
        "Healthy": [0.7, 0.4],
        "Fever": [0.3, 0.6]
    },
    index=hidden_states)

emit_prob_df = DataFrame(  # Probability of symptom emission
    data={ # Normal when healthy, normal when 
        "Normal": [0.5, 0.1], # fever, etc.
        "Cold": [0.4, 0.3],
        "Dizzy": [0.1, 0.6]
    },
    index=hidden_states)

### Initialize starting probabilities
start_probs = {"Healthy": 0.6, "Fever": 0.4}

### Dynammic Programming Matrix
viterbi_df = DataFrame(index=hidden_states)

### Initialize Probability 0
for state in hidden_states:  # For each state
    viterbi_df.loc[  # Multiple the starting probability by the corresponding
        state,  # emission probability for the initial observation
        "Probability 0"] = start_probs[state] * emit_prob_df.loc[state][observations[0]]

### Begin Dynamic Programming
for i in range(1, len(observations)):  # For the given observations
    for state in hidden_states:  # For each current state
        # Calculate all transition probabilities and take the max
        max_trans_prob = max(viterbi_df.iloc[:, i - 1][prev_state] *
                             trans_prob_df.loc[prev_state][state]
                             for prev_state in hidden_states)
        for prev_state in hidden_states:  # For each previous state
            # Multiply the max transition probability by each emission probability,
            # for the correct state and observation
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
    " with a probability of %s" % max_resulting_prob + "\n")

### Print dynammic programming matrix
print(viterbi_df)  # DONE!
