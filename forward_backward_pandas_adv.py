from pandas import DataFrame
from itertools import cycle

# Initialize tuples of conditions.  Observations are the input
observations = ("Wearing Trenchcoat & Fedora", "Eating Pizza",
                "Eating Doritos", "Browsing Reddit", "Playing WoW", "Smelly",
                "Vaping", "Listening to Power Metal", "Brandishing Katana",
                "Wearing Trenchcoat & Fedora", "Browsing 4chan",
                "Playing Magic the Gathering", "Drinking Mountain Dew")
hidden_states = ("Depressed", "Confident", "Tired", "Hungry", "Thirsty",
                 "Angry", "Gamer", "Brony", "Libertarian", "Atheist", "End")
emit_states = ("Eating Pizza", "Browsing Reddit", "Drinking Mountain Dew",
               "Eating Doritos", "Wearing Trenchcoat & Fedora",
               "Browsing 4chan", "Playing Magic the Gathering", "Playing WoW",
               "Brandishing Katana", "Watching My Little Pony",
               "Listening to Power Metal", "Vaping", "Smelly")

# Color pattern for final output
colors = cycle(["red", "orange", "green", "blue", "purple"])  # Rainbow effect
colors_dict = {}  # Each observation gets a color

# Probability of transition from state to state, remaining static, etc.
trans_prob_df = DataFrame(
    data={
        "Depressed": (0.10, 0.10, 0.10, 0.15, 0.05, 0.15, 0.05, 0.05, 0.05,
                      0.05),
        "Confident": (0.05, 0.05, 0.05, 0.10, 0.10, 0.14, 0.05, 0.15, 0.10,
                      0.10),
        "Tired": (0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.05, 0.05, 0.05),
        "Hungry": (0.14, 0.05, 0.10, 0.14, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05),
        "Thirsty": (0.10, 0.14, 0.15, 0.10, 0.10, 0.05, 0.05, 0.10, 0.05,
                    0.05),
        "Angry": (0.05, 0.10, 0.14, 0.05, 0.15, 0.15, 0.15, 0.05, 0.15, 0.15),
        "Gamer": (0.10, 0.10, 0.10, 0.10, 0.14, 0.10, 0.19, 0.15, 0.05, 0.05),
        "Brony": (0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.15, 0.29, 0.05, 0.05),
        "Libertarian": (0.10, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05, 0.29,
                        0.15),
        "Atheist": (0.10, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05, 0.15,
                    0.29),
        "End": (0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
    },
    columns=hidden_states,
    index=hidden_states[:-1])  # All should vertically sum to 1

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
    index=hidden_states[:-1])  # All should vertically sum to 1

# Initialize starting probabilities
start_probs = DataFrame(
    data={
        "(0) {}".format(observations[0]): (0.10, 0.10, 0.10, 0.15, 0.10, 0.10,
                                           0.15, 0.10, 0.05, 0.05)
    },
    index=hidden_states[:-1])

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
