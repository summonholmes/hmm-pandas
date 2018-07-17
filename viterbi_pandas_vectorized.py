"""Based off of https://en.wikipedia.org/wiki/Viterbi_algorithm"""
from pandas import DataFrame, Series
from numpy import select, where, max as npmax
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
    max_trans_prob_df = trans_prob_df.multiply(
        viterbi_df.iloc[:, i - 1], axis="index").apply(
            npmax, axis=0)  # Vectorize maximums for the previous column
    viterbi_df["Probability {}".format(
        i)] = max_trans_prob_df * emit_prob_df.loc[:, observations[i]]

    max_resulting_prob = viterbi_df.iloc[:, -1].max()
dyn_prog_path = [viterbi_df.iloc[:, -1].idxmax()]

### Traceback (Manual Recursive Demostration, using known sequence)
where(viterbi_df.iloc[:, 3] * trans_prob_df.loc[:, "Confident"] *
      emit_prob_df.loc["Confident", "Wearing Trenchcoat & Fedora"] ==
      max_resulting_prob)[0][0]
where(viterbi_df.iloc[:, 2] * trans_prob_df.loc[:, "Thirsty"] *
      emit_prob_df.loc["Thirsty", "Eating Doritos"] == viterbi_df.iloc[4, 3])[
          0][0]  # Done
where(viterbi_df.iloc[:, 1] * trans_prob_df.loc[:, "Tired"] *
      emit_prob_df.loc["Tired", "Drinking Mountain Dew"] == viterbi_df.iloc[
          2, 2])[0][0]  # Done
where(viterbi_df.iloc[:, 0] * trans_prob_df.loc[:, "Tired"] *
      emit_prob_df.loc["Tired", "Browsing Reddit"] == viterbi_df.iloc[2, 1])[
          0][0]  # Done
where(viterbi_df.iloc[:, 0] == viterbi_df.iloc[3, 0])[0][0]  # Done
 
### Print dynammic programming matrix
viterbi_df.style.background_gradient(
    cmap=light_palette("green", as_cmap=True))  # DONE!
