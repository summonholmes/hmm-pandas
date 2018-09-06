from pandas import DataFrame, IndexSlice
from seaborn import light_palette

# Initialize tuples of conditions.  Observations are the input
observations = ("Wearing Trenchcoat & Fedora", "Eating Pizza",
                "Eating Doritos", "Browsing Reddit", "Playing WoW", "Smelly",
                "Vaping", "Listening to Power Metal", "Brandishing Katana",
                "Wearing Trenchcoat & Fedora", "Browsing 4chan",
                "Playing Magic the Gathering", "Drinking Mountain Dew")
hidden_states = ("Depressed", "Confident", "Tired", "Hungry", "Thirsty",
                 "Angry", "Gamer", "Brony", "Libertarian", "Atheist")
emit_states = ("Eating Pizza", "Browsing Reddit", "Drinking Mountain Dew",
               "Eating Doritos", "Wearing Trenchcoat & Fedora",
               "Browsing 4chan", "Playing Magic the Gathering", "Playing WoW",
               "Brandishing Katana", "Watching My Little Pony",
               "Listening to Power Metal", "Vaping", "Smelly")

# Probability of transition from state to state, remaining static, etc.
trans_prob_df = DataFrame(
    data={
        "Depressed": (0.10, 0.10, 0.10, 0.15, 0.05, 0.15, 0.05, 0.05, 0.05,
                      0.05),
        "Confident": (0.05, 0.05, 0.05, 0.10, 0.10, 0.15, 0.05, 0.15, 0.10,
                      0.10),
        "Tired": (0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.05, 0.05, 0.05),
        "Hungry": (0.15, 0.05, 0.10, 0.15, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05),
        "Thirsty": (0.10, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05, 0.10, 0.05,
                    0.05),
        "Angry": (0.05, 0.10, 0.15, 0.05, 0.15, 0.15, 0.15, 0.05, 0.15, 0.15),
        "Gamer": (0.10, 0.10, 0.10, 0.10, 0.15, 0.10, 0.20, 0.15, 0.05, 0.05),
        "Brony": (0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.15, 0.30, 0.05, 0.05),
        "Libertarian": (0.10, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05, 0.30,
                        0.15),
        "Atheist": (0.10, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05, 0.15, 0.30)
    },
    columns=hidden_states,
    index=hidden_states)  # All should vertically sum to 1

# Probability of observation given the hidden state
emit_prob_df = DataFrame(
    data={
        "Eating Pizza": (0.10, 0.05, 0.05, 0.20, 0.05, 0.05, 0.05, 0.05, 0.05,
                         0.05),
        "Browsing Reddit": (0.10, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05,
                            0.15, 0.15),
        "Drinking Mountain Dew": (0.10, 0.10, 0.15, 0.10, 0.20, 0.05, 0.15,
                                  0.05, 0.05, 0.05),
        "Eating Doritos": (0.10, 0.05, 0.05, 0.15, 0.10, 0.05, 0.10, 0.05,
                           0.05, 0.05),
        "Wearing Trenchcoat & Fedora": (0.05, 0.20, 0.05, 0.05, 0.05, 0.05,
                                        0.05, 0.10, 0.10, 0.10),
        "Browsing 4chan": (0.10, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05, 0.10,
                           0.05, 0.05),
        "Playing Magic the Gathering": (0.05, 0.05, 0.05, 0.05, 0.05, 0.10,
                                        0.15, 0.05, 0.05, 0.05),
        "Playing WoW": (0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.15, 0.05, 0.05,
                        0.05),
        "Brandishing Katana": (0.05, 0.10, 0.05, 0.05, 0.05, 0.10, 0.05, 0.05,
                               0.15, 0.15),
        "Watching My Little Pony": (0.05, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05,
                                    0.25, 0.05, 0.10),
        "Listening to Power Metal": (0.05, 0.05, 0.05, 0.05, 0.10, 0.15, 0.05,
                                     0.05, 0.05, 0.10),
        "Vaping": (0.05, 0.10, 0.05, 0.05, 0.10, 0.05, 0.05, 0.05, 0.10, 0.05),
        "Smelly": (0.15, 0.10, 0.15, 0.10, 0.10, 0.15, 0.05, 0.10, 0.10, 0.05)
    },
    columns=emit_states,
    index=hidden_states)  # All should vertically sum to 1

# Initialize starting probabilities
start_probs = DataFrame(
    data={
        "(0) {}".format(observations[0]): (0.10, 0.10, 0.10, 0.15, 0.10, 0.10,
                                           0.15, 0.10, 0.05, 0.05)
    },
    index=hidden_states)

# Initialize dynammic programming matrix at probability 0
viterbi_df = start_probs.multiply(emit_prob_df[observations[0]], axis="index")

# Start dynammic programming
for i, observation in enumerate(observations[1:]):
    max_trans_prob_df = trans_prob_df.multiply(  # Offset by 1
        viterbi_df.iloc[:, i], axis="index").max()
    # Multiply entire trans_prob df by previous viterbi_df
    # column and take vertical maximums
    viterbi_df["({}) {}".format(
        i + 1,  # Then multiply the result by the observation emissions
        observation)] = max_trans_prob_df * emit_prob_df.loc[:, observation]

# Provide the entire matrix with highest values darkest
viterbi_traceback_df = viterbi_df.style.background_gradient(
    cmap=light_palette("green", as_cmap=True))

# At the last column, use the maximum value to begin traceback
traceback_prob = [viterbi_df.iloc[:, -1].max()]
dyn_prog_path = [viterbi_df.iloc[:, -1].idxmax()]  # And its index
viterbi_traceback_df.highlight_max(  # Highlight it
    color="red", subset=IndexSlice[[viterbi_df.columns[-1]]])

# Start traceback
for i, observation in zip(  # Reverse enumerate with offset
        range(len(observations) - 2, -1, -1), reversed(observations[1:])):
    # Isolate the previous location that gives the current probability
    traceback_loc = viterbi_df.loc[  # Always going left-most
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

# Print dynammic programming matrix and traceback results
print("The observations:", ", ".join(observations))
print("The sequence of hidden states is most likely:")
print((viterbi_df.isin(traceback_prob)).idxmax())
print("The final probability:", traceback_prob[-1])
viterbi_traceback_df
