import numpy as np
import matplotlib.pyplot as plt

state: np.ndarray = np.load("data/chat-model/state.npy").reshape(24, -1, 2048)
state = state[::, 2::, ::]
state = state.reshape(24, 32, 64, 64)
mean = state.mean(axis=(2, 3)).reshape(24, 32, 1, 1)
std = state.std(axis=(2, 3)).reshape(24, 32, 1, 1)
state = (state - mean) / (std / 2)
state = np.pad(state, ((0, 0), (0, 0), (1, 1), (1, 1)), "constant", constant_values=(float("nan"), float("nan")))
state = state.transpose(0, 2, 1, 3)
state = state.reshape(24 * 66, 32 * 66)
# state = np.log(1 + state.clip(min=0)) - np.log(1 - state.clip(max=0))
state = np.tanh(state)
# state = np.clip(np.cbrt(state), -10, 10)

ax = plt.matshow(state)
plt.colorbar(ax.colorbar)
plt.show()
