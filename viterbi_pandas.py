from pandas import DataFrame

### Initialize tuples of conditions
observations = ("Normal", "Cold", "Dizzy"
                )  # Calculate the probability for this sequence of events
hidden_states = ("Healthy", "Fever")  # The confounding factors

### Initialize all dataframes
trans_prob_df = DataFrame(  # Probability of transition
    data={
        "Healthy": [0.7, 0.4],
        "Fever": [0.3, 0.6]
    },
    index=hidden_states)

# Possibilities are:
# Healthy->Healthy = 0.7
# Healthy->Fever = 0.4
# Fever->Fever = 0.3
# Fever->Healthy = 0.6

emit_prob_df = DataFrame(  # Probability of symptom emission
    data={
        "Normal": [0.5, 0.1],
        "Cold": [0.4, 0.3],
        "Dizzy": [0.1, 0.6]
    },
    index=hidden_states)

# Possibilities are:
# Healthy==Normal = 0.5
# Healthy==Cold = 0.4
# Healthy==Dizzy = 0.1
# Fever==Normal = 0.1
# Fever==Cold = 0.3
# Fever==Dizzy = 0.6

viterbi_df = DataFrame(index=hidden_states)  # Dynammic Programming Matrix

### Initialize starting probabilities
start_probs = {"Healthy": 0.6, "Fever": 0.4}

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
            # Multiple the max transition probability by each emission probability
            # for the correct state and observation
            max_prob = max_trans_prob * emit_prob_df.loc[state][observations[i]]
            viterbi_df.loc[state, "Probability {}".format(i)] = max_prob

# Obtain the highest possible probability for the problem
max_resulting_prob = viterbi_df.iloc[:, -1].max()
