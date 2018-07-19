from pandas import DataFrame
from numpy import max as npmax, sum as npsum

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
        "Tired": (0.24, 0.09, 0.29, 0.10, 0.15),
        "Hungry": (0.20, 0.20, 0.25, 0.29, 0.29),
        "Thirsty": (0.20, 0.20, 0.25, 0.20, 0.10),
        "End": (0.01, 0.01, 0.01, 0.01, 0.01)
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

### Initialize forward dataframe
forward_df = start_probs.multiply(emit_prob_df[observations[0]], axis="index")

### Start forward part
for i, observation in enumerate(observations[1:]):
    previous_forward_sum = trans_prob_df.iloc[:, :-1].multiply(
        forward_df.iloc[:, i], axis="index").apply(npsum)
    forward_df["Probability {}".format(
        i + 1)] = previous_forward_sum * emit_prob_df.loc[:, observation]

### Calculate forward probability
forward_prob = (forward_df.iloc[:, -1] * trans_prob_df.iloc[:, -1]).sum()
