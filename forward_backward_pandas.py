from pandas import DataFrame
from itertools import cycle

### Initialize tuples of conditions.  Observations are the input
observations = ("Wearing Trenchcoat & Fedora", "Browsing Reddit",
                "Drinking Mountain Dew", "Eating Doritos", "Eating Pizza")
hidden_states = ("Depressed", "Confident", "Tired", "Hungry",
                 "Thirsty")  # The confounding factors
colors = cycle(["red", "orange", "green", "blue", "purple"])
colors_dict = {}  # To allow larger observation sequences

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
    data={"(0) {}".format(observations[0]): (0.10, 0.40, 0.10, 0.20, 0.20)},
    index=hidden_states)

### Initialize forward dataframe
forward_df = start_probs.multiply(emit_prob_df[observations[0]], axis="index")
colors_dict[forward_df.columns[0]] = next(colors)

### Start forward part - 1st pass
for i, observation in enumerate(observations[1:]):
    previous_forward_sum = trans_prob_df.iloc[:, :-1].multiply(
        forward_df.iloc[:, i], axis="index").sum()
    forward_df["({}) {}".format(
        i + 1,  # Very similar to Viterbi, but sum
        observation)] = previous_forward_sum * emit_prob_df.loc[:, observation]
    colors_dict[forward_df.columns[i + 1]] = next(colors)

### Calculate forward probability
forward_prob = (forward_df.iloc[:, -1] * trans_prob_df.iloc[:, -1]).sum()

### Initialize backward dataframe
backward_df = DataFrame(
    data={
        "({}) {}".format(len(observations) - 1, observations[-1]):
        trans_prob_df.iloc[:, -1]
    })

### Start backward part - 2nd pass
for i, observation in zip(
        range(len(observations) - 2, -1, -1), reversed(observations[1:])):
    backward_df.insert(  # Similar to forward and Viterbi traceback
        0,  # Countdown to 2nd observation
        "({}) {}".format(i, observations[i]),
        (backward_df.iloc[:, 0] * trans_prob_df.iloc[:, :-1] *
         emit_prob_df.loc[:, observation]).sum(axis=1))

### Calculate backward probability: Should == forward probability
backward_prob = (backward_df.iloc[:, 0] * start_probs.iloc[:, 0] *
                 emit_prob_df.loc[:, observations[0]]).sum()

### Now merge the two
posterior_df = (forward_df * backward_df).apply(lambda x: x / forward_prob)

### Stylized output for reading top-down
posterior_df_style = posterior_df.style.apply(  # Color the columns
    lambda x: ["background-color: {}".format(colors_dict[x.name])] * len(x))

### Print final results
print("The observations:", ", ".join(observations))
print("The summed forward probability:", forward_prob)
print("The summed backward probability:", backward_prob)
print("Posterior marginals are read top-down")
posterior_df_style.highlight_max(color="black")  # Highlight maximums
