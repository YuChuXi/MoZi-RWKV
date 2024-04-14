import numpy as np
import matplotlib.pyplot as plt
from rwkv import RWKVState


def show_att_state(model, state: np.ndarray, norm=True, only_std=False):
    # 恢复形状
    n_embed = model.n_embed
    n_layer = model.n_layer
    head_size = state.size // model.n_layer // n_embed -2
    n_head = n_embed // head_size

    state = state.reshape(n_layer, -1, n_embed)
    state = state[::, 2::, ::]
    state = state.reshape(n_layer, n_head, head_size, head_size)

    if norm or only_std:
        std = state.std(axis=(2, 3))
    if only_std:
        ax = plt.matshow(std, norm="log")
        plt.colorbar(ax.colorbar)
        plt.pause(2)
        plt.close()
        return
    if norm:  # 归一化，state的mean是0
        state = state / (std.reshape(n_layer, n_head, 1, 1) / 2)

    state = np.pad(
        state,
        ((0, 0), (0, 0), (1, 1), (1, 1)),
        "constant",
        constant_values=(float("nan"), float("nan")),
    )
    state = state.transpose(0, 2, 1, 3)
    state = state.reshape(n_layer * 66, n_head * 66)
    ax = plt.matshow(state, norm="asinh")
    plt.colorbar(ax.colorbar)
    plt.pause(2)
    plt.close()


def show_xx_state(model, state: np.ndarray):
    n_embed = model.n_embed
    n_layer = model.n_layer
    head_size = state.size // model.n_layer // n_embed -2
    n_head = n_embed // head_size

    state = state.reshape(n_layer, -1, n_embed)
    state = state[::, :2:, ::]
    state = state.reshape(n_layer * 2, n_embed)
    ax = plt.matshow(state)
    plt.colorbar(ax.colorbar)
    plt.pause(2)
    plt.close()


class show_state_delta:
    def __init__(self, model, state: RWKVState) -> None:
        self.model = model
        self.state: RWKVState = state
        self.state_cache: np.ndarray = None

    def __enter__(self):
        self.state_cache = self.state.state.copy()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        show_att_state(self.model, self.state.state - self.state_cache, only_std=True)
        del self.state_cache

