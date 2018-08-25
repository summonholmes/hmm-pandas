from pandas import DataFrame
from itertools import cycle

# Initialize tuples of conditions.  Observations are the input
observations = (  # Modify, add, remove with any key in emit_prob_df
    "Wearing Trenchcoat & Fedora", "Browsing Reddit", "Drinking Mountain Dew",
    "Eating Doritos", "Eating Pizza")
hidden_states = ("Depressed", "Confident", "Tired", "Hungry",
                 "Thirsty")  # The confounding factors
colors = cycle(["red", "orange", "green", "blue", "purple"])  # Rainbow effect
colors_dict = {}  # Each observation gets a color

# Probability of transition
trans_prob_df = DataFrame(
    data={  # From depressed to confident, vice-versa, static, etc.
        "Depressed": (0.20, 0.25, 0.10, 0.20, 0.20),
        "Confident": (0.15, 0.25, 0.10, 0.20, 0.25),
        "Tired": (0.24, 0.09, 0.29, 0.10, 0.15),
        "Hungry": (0.20, 0.20, 0.25, 0.29, 0.29),
        "Thirsty": (0.20, 0.20, 0.25, 0.20, 0.10),
        "End": (0.01, 0.01, 0.01, 0.01, 0.01)
    },  # All should vertically sum to 1
    columns=("Depressed", "Confident", "Tired", "Hungry", "Thirsty", "End"),
    index=hidden_states)

# Probability of observation given the hidden state
emit_prob_df = DataFrame(
    data={  # Highest chance of trenchcoat & fedora is when confident
        "Eating Pizza": (0.20, 0.10, 0.10, 0.35, 0.20),
        "Browsing Reddit": (0.20, 0.10, 0.35, 0.10, 0.20),
        "Drinking Mountain Dew": (0.30, 0.10, 0.30, 0.20, 0.30),
        "Eating Doritos": (0.20, 0.10, 0.15, 0.15, 0.15),
        "Wearing Trenchcoat & Fedora": (0.10, 0.60, 0.10, 0.20, 0.15),
    },  # All should vertically sum to 1
    columns=("Eating Pizza", "Browsing Reddit", "Drinking Mountain Dew",
             "Eating Doritos", "Wearing Trenchcoat & Fedora"),
    index=hidden_states)

# Initialize starting probabilities
start_probs = DataFrame(
    data={"(0) {}".format(observations[0]): (0.10, 0.40, 0.10, 0.20, 0.20)},
    index=hidden_states)

# Initialize forward dataframe
forward_df = start_probs.multiply(emit_prob_df[observations[0]], axis="index")
colors_dict[forward_df.columns[0]] = next(colors)  # For final colored output

# Start forward part - 1st pass
for i, observation in enumerate(observations[1:]):  # Same as viterbi
    previous_forward_sum = trans_prob_df.iloc[:, :-1].multiply(
        forward_df.iloc[:, i], axis="index").sum()
    forward_df["({}) {}".format(
        i + 1,  # Similar to Viterbi but sum, line below is identical
        observation)] = previous_forward_sum * emit_prob_df.loc[:, observation]
    colors_dict[forward_df.columns[i + 1]] = next(colors)  # Update colors

# Calculate forward probability
# Multiply last columns and sum the result
forward_prob = (forward_df.iloc[:, -1] * trans_prob_df.iloc[:, -1]).sum()

# Initialize backward dataframe
backward_df = DataFrame(
    data={  # The last column of trans_prob_df
        "({}) {}".format(len(observations) - 1, observations[-1]):
        trans_prob_df.iloc[:, -1]
    })

# Start backward part - 2nd pass
for i, observation in zip(  # Same as viterbi
        range(len(observations) - 2, -1, -1), reversed(observations[1:])):
    backward_df.insert(  # Countdown to 2nd observation
        0,  # The left-most column updates itself by multiplying
        # The entire trans_prob_df and emit_prob_df that matches observation
        "({}) {}".format(i, observations[i]),
        (backward_df.iloc[:, 0] * trans_prob_df.iloc[:, :-1] *
         emit_prob_df.loc[:, observation]).sum(axis=1))  # Horizontal sum

# Calculate backward probability: Should == forward probability
# Now use beginning values, opposite of forward
backward_prob = (backward_df.iloc[:, 0] * start_probs.iloc[:, 0] *
                 emit_prob_df.loc[:, observations[0]]).sum()

# Now merge the two - vectorized multiplication of all and divide by either
# forward or backward probability
posterior_df = (forward_df * backward_df) / forward_prob

# Stylized output for reading top-down
posterior_df_style = posterior_df.style.apply(  # Color the columns
    lambda x: ["background-color: {}".format(colors_dict[x.name])] * len(x))

# Print final results - table should vertically sum to 1
print("The observations:", ", ".join(observations))
print("The most likely non-sequential hidden states are:")
print(posterior_df.idxmax())
print("The summed forward & backward probabilities: ", forward_prob, ",",
      backward_prob)
posterior_df_style.highlight_max(color="black")  # Highlight maximums
