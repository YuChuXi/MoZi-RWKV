import numpy as np
import matplotlib.pyplot as plt

state: np.ndarray = np.load("data/chat-model/state.npy").reshape(24, -1, 2048)
state = state[::, 2::, ::]
state = state.reshape(24, 32, 64, 64)
state = np.pad(state, ((0, 0), (0, 0), (1, 1), (1, 1)), "constant", constant_values=(1e-6, 1e-6))
state = state.transpose(0, 2, 1, 3)
state = state.reshape(24 * 66, 32 * 66)
# state = np.log10(1 + state.clip(min=0)) - np.log10(1 - state.clip(max=0))
state = np.tanh(state)

ax = plt.matshow(state)
plt.colorbar(ax.colorbar)
plt.show()
