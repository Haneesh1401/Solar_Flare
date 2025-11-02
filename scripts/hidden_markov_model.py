import numpy as np
from hmmlearn import hmm

# 1. Define the HMM parameters

# Hidden States: Mood (0=Happy, 1=Sad)
states = ["Happy", "Sad"]
n_states = len(states)

# Possible Observations: Activity (0=Shop, 1=Walk)
observations = ["Shop", "Walk"]
n_observations = len(observations)
# Mapping the observations to indices
obs_map = dict(zip(observations, range(n_observations)))

# Initial Probabilities (startprob_): P(state at t=0)
start_probability = np.array([0.6, 0.4]) # Starts Happy (0.6) or Sad (0.4)

# Transition Probabilities (transmat_): P(state_t | state_{t-1})
# Rows are previous state, columns are current state
# P(Happy|Happy) = 0.7, P(Sad|Happy) = 0.3
# P(Happy|Sad) = 0.4, P(Sad|Sad) = 0.6
transition_probability = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

# Emission Probabilities (emissionprob_): P(observation | state)
# Rows are states, columns are observations
# Happy: P(Shop|Happy) = 0.8, P(Walk|Happy) = 0.2
# Sad: P(Shop|Sad) = 0.3, P(Walk|Sad) = 0.7
emission_probability = np.array([
    [0.8, 0.2],
    [0.3, 0.7]
])

# 2. Create and initialize the HMM model
model = hmm.MultinomialHMM(n_components=n_states, random_state=42, n_iter=100, n_trials=1)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

# 3. Predict the most likely sequence of hidden states (Viterbi Algorithm)

# Observed sequence: "Shop", "Walk", "Shop", "Walk"
observed_sequence_list = ["Shop", "Walk", "Shop", "Walk"]
# Convert observations to one-hot encoded vectors for MultinomialHMM
observed_sequence_indices = np.zeros((len(observed_sequence_list), n_observations))
for i, obs in enumerate(observed_sequence_list):
    idx = obs_map[obs]
    observed_sequence_indices[i, idx] = 1

# Predict the hidden states using the Viterbi algorithm
# score: The log likelihood of the observation sequence given the model
# predicted_states: The most likely sequence of states
log_likelihood, predicted_states = model.decode(observed_sequence_indices, algorithm='viterbi')

# Convert predicted state indices back to state names
predicted_state_names = [states[i] for i in predicted_states]

print("--- HMM (Viterbi Decoding) ---")
print(f"Observed Sequence: {observed_sequence_list}")
print(f"Most Likely Hidden State Sequence: {predicted_state_names}")
print(f"Log Likelihood of the sequence: {log_likelihood:.4f}")