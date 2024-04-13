import numpy as np
import matplotlib.pyplot as plt
from rwkv import RWKVState


def show_att_state(state: np.ndarray, norm=True, only_std=False):
    # 恢复形状
    state = state.reshape(24, -1, 2048)
    state = state[::, 2::, ::]
    state = state.reshape(24, 32, 64, 64)

    if norm or only_std:
        std = state.std(axis=(2, 3))
    if only_std:
        ax = plt.matshow(np.log10(std))
        plt.colorbar(ax.colorbar)
        plt.show()
        return
    if norm:  # 归一化，state的mean是0
        state = state / (std.reshape(24, 32, 1, 1) / 2)

    state = np.pad(
        state,
        ((0, 0), (0, 0), (1, 1), (1, 1)),
        "constant",
        constant_values=(float("nan"), float("nan")),
    )
    state = state.transpose(0, 2, 1, 3)
    state = state.reshape(24 * 66, 32 * 66)
    state = np.log10(1 + state.clip(min=0)) - np.log10(1 - state.clip(max=0))
    # state = np.tanh(state)
    # state = np.clip(np.cbrt(state), -10, 10)

    ax = plt.matshow(state)
    plt.colorbar(ax.colorbar)
    plt.show()


def show_xx_state(state: np.ndarray):
    state = state.reshape(24, -1, 2048)
    state = state[::, :2:, ::]
    state = state.reshape(48, 2048)
    ax = plt.matshow(state)
    plt.colorbar(ax.colorbar)
    plt.show()


class show_state_delta:
    def __init__(self, state: RWKVState) -> None:
        self.state: RWKVState = state
        self.state_cache: np.ndarray = None

    def __enter__(self):
        self.state_cache = self.state.state.copy()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        show_att_state(self.state.state - self.state_cache, only_std=True)
        del self.state_cache
