# Provides terminal-based chat interface for RWKV model.
# Usage: python chat_with_bot.py C:\rwkv.cpp-169M.bin
# Prompts and code adapted from https://github.com/Blink/bloEmbryo:/9ca4cdba90efaee25cfec21a0bae72cbd48d8acd/chat.py

import os
import pickle
import json
import tqdm
import time
import numpy as np
import sampling
import multiprocessing
from threading import Lock
from rwkv_cpp import rwkv_cpp_shared_library, rwkv_cpp_model
from rwkv_cpp.rwkv_world_tokenizer import RWKV_TOKENIZER

# from tokenizer_util import get_tokenizer
from typing import List, Dict, Optional
from app_util import prxxx, check_dir

# ======================================== Script settings ========================================

# English, Chinese, Japanese
LANGUAGE: str = "Chinese"
# QA: Question and Answer prompt to talk to an AI assistant.
# Chat: chat prompt (need a large model for adequate quality, 7B+).
PROMPT_TYPE: str = "Chat-MoZi-N"

MAX_GENERATION_LENGTH: int = 128

# Sampling temperature. It could be a good idea to increase temperature when top_p is low.
TEMPERATURE: float = 1.0
# For better Q&A accuracy and less diversity, reduce top_p (to 0.5, 0.2, 0.1 etc.)
TOP_P: float = 0.7
# Penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
PRESENCE_PENALTY: float = 1.7
# Penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
FREQUENCY_PENALTY: float = 2.1

# END_OF_LINE_TOKEN: int = 187
# DOUBLE_END_OF_LINE_TOKEN: int = 535
END_OF_TEXT_TOKEN: int = 0

np.random.seed(int(time.time() * 1e6 % 2**30))

model_name = "RWKV-5-World-3B-Q4_0-v2l"
model_name = "RWKV-5-Qun-1B5-Q4_0"
model_name = "RWKV-5-World-1B5-Q5_1-v2"
model_path = f"model/{model_name}.bin"

model_state_name = "default.state"
model_state_path = f"data/{model_state_name}.pkl"

tokenizer_dict = "rwkv_cpp/rwkv_vocab_v20230424.txt"

# =================================================================================================

library = rwkv_cpp_shared_library.load_rwkv_shared_library()
prxxx(f"System info: {library.rwkv_get_system_info_string()}")

prompt_config = f"prompt/{LANGUAGE}-{PROMPT_TYPE}.json"
prxxx(f"Loading RWKV prompt config: {prompt_config}")
with open(prompt_config, "r", encoding="utf8") as json_file:
    prompt_data = json.load(json_file)
    user, bot, separator, default_init_prompt = (
        prompt_data["user"],
        prompt_data["bot"],
        prompt_data["separator"],
        prompt_data["prompt"],
    )
    if os.path.isfile(default_init_prompt):
        with open(default_init_prompt, "rb") as f:
            default_init_prompt = f.read().decode("utf-8")
assert default_init_prompt != "", "Prompt must not be empty"


prxxx(f"Loading RWKV model: {model_path}")
model = rwkv_cpp_model.RWKVModel(library, model_path, thread_count=10)
model_lock = Lock()

check_dir("data")
if os.path.isfile(f"data/tokenizer.pkl"):
    prxxx(f"Loading tokenizer: data/tokenizer.pkl")
    with open(f"data/tokenizer.pkl", "rb") as f:
        tokenizer: RWKV_TOKENIZER = pickle.load(f)
else:
    prxxx(f"Loading tokenizer: {tokenizer_dict}")
    tokenizer: RWKV_TOKENIZER = RWKV_TOKENIZER(tokenizer_dict)
    with open(f"data/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

# =================================================================================================


class RWKVEmbryo:
    def __init__(self, id: str, state_name: str = model_state_name, prompt: str = None):
        prxxx(
            f"Init RWKV id:{id} state: {state_name} prompt: {'None' if prompt is None else prompt.strip().splitlines()[0]}"
        )
        self.id: str = str(id)
        check_dir(f"data/{id}")

        self.default_state: str = state_name
        self.logits: np.ndarray = None
        self.state: np.ndarray = None
        self.processed_tokens: List[int] = []
        self.processed_tokens_counts: Dict[int, int] = {}
        self.process_lock: Lock = Lock()
        self.mlog = open(f"data/{self.id}/model.log", "ab+")
        self.ulog = open(f"data/{self.id}/user.log", "a+")
        self.presence_penalty: float = PRESENCE_PENALTY
        self.frequency_penalty: float = FREQUENCY_PENALTY

        self.load_state(self.id, prompt)

    def __del__(self):
        self.mlog.close()
        self.ulog.close()

    def load_state(self, state_name: str, prompt: str = None):
        if prompt is not None:
            prompt_tokens = tokenizer.encode(prompt)
            prxxx(f"Process prompt tokens, length: {len(prompt_tokens)} tok")
            ltime = time.time()
            self.process_tokens(prompt_tokens)
            prxxx(f"Processed prompt tokens, used: {int(time.time()-ltime)} s")
            self.save_state(self.default_state)
            prxxx(f"Save state: {self.default_state}")
            self.mlog.write(f" : Load[{state_name}]\n\n".encode("utf-8"))
        else:
            state_names = [self.default_state, model_state_name]
            if state_name is not None:
                state_names = [state_name] + state_names

            for state_name in state_names:
                if not os.path.isfile(f"data/{state_name}/tokens.pkl"):
                    continue

                prxxx(f"Load state: {state_name}")
                with open(f"data/{state_name}/tokens.pkl", "rb") as f:
                    data = pickle.load(f)
                    self.processed_tokens: List[int] = data["processed_tokens"]
                    self.logits: np.ndarray = data["logits"]
                    
                    self.processed_tokens_counts: Dict[int, int] = data[
                        "processed_tokens_counts"
                    ]
                self.state: np.ndarray = np.load(f"data/{state_name}/state.npy")

            self.mlog.write(f" : Load[{state_name}]\n\n".encode("utf-8"))
        self.mlog.flush()

    def save_state(self, state_name: str):
        check_dir(f"data/{state_name}")
        with open(f"data/{state_name}/tokens.pkl", "wb") as f:
            pickle.dump(
                {
                    "processed_tokens": self.processed_tokens,
                    "logits": self.logits,
                    "processed_tokens_counts": self.processed_tokens_counts,
                },
                f,
            )
        np.save(f"data/{state_name}/state.npy",self.state)
        self.mlog.flush()
        self.ulog.flush()

    def reset(self):
        self.load_state(self.default_state)
        self.ulog.write(" : Reset")
        self.save_state(self.id)

    def check_state(self):
        return
        l = self.logits
        s = self.state
        if "numpy" in dir(s):
            l = l.numpy()
            s = s.numpy()
        s_var = s.var()
        prxxx(
            "*  logits:\tmean\t%.2f\tvar\t%.2f\tmax\t%.2f\tmin %.2f"
            % (l.mean(), l.var(), l.max(), l.min())
        )
        prxxx(
            "*  state:\tmean\t%.2f\tvar\t%.2f\tmax\t%.2f\tmin %.2f"
            % (s.mean(), s_var, s.max(), s.min())
        )
        prxxx(
            "*  san:\t%.3f" % (10 - np.log(s_var) / 0.6214608098422),
            "" if s_var < 500 else "!",
        )
        # self.presence_penalty = s_var/72
        # self.frequency_penalty = s_var/36

    def process_processed_tokens_counts(self, token: int):
        self.processed_tokens += [token]
        for token in self.processed_tokens_counts:
            self.processed_tokens_counts[token] *= 0.99

        if token not in self.processed_tokens_counts:  # 词频统计
            self.processed_tokens_counts[token] = 1
        else:
            self.processed_tokens_counts[token] += 1

    def process_token_penalty(self):
        for token in self.processed_tokens_counts:
            self.logits[token] -= (
                self.presence_penalty
                + self.processed_tokens_counts[token] * self.frequency_penalty
            )

    def process_tokens(
        self, tokens: List[int], new_line_logit_bias: float = 0.0
    ) -> None:
        """
        self.logits, self.state = model.eval_sequence(
            tokens, self.state, self.state, self.logits, use_numpy=True)
        self.processed_tokens += tokens
        #self.logits[END_OF_LINE_TOKEN] += new_line_logit_bias
        """

        for token in tqdm.tqdm(
            tokens, desc="Processing prompt", leave=False, unit=" tok"
        ):
            with model_lock:
                self.logits, self.state = model.eval(
                    token, self.state, self.state, self.logits
                )
            self.process_processed_tokens_counts(token)
            self.check_state()
        # self.logits[END_OF_LINE_TOKEN] += new_line_logit_bias
        self.mlog.write(tokenizer.decodeBytes(tokens))

    def process_token(self, token: int, new_line_logit_bias: float = 0.0) -> None:
        with model_lock:
            self.logits, self.state = model.eval(
                token, self.state, self.state, self.logits
            )

        self.process_processed_tokens_counts(token)
        # self.logits[END_OF_LINE_TOKEN] += new_line_logit_bias
        self.check_state()
        self.mlog.write(tokenizer.decodeBytes([token]))

    """
    # Model only saw '\n\n' as [187, 187] before, but the tokenizer outputs [535] for it at the end.
    # See https://github.com/Blink/pulEmbryo:/110/files


    def split_last_end_of_line(self, tokens: List[int]) -> List[int]:
        if len(tokens) > 0 and tokens[-1] == DOUBLE_END_OF_LINE_TOKEN:
            tokens = tokens[:-1] + [END_OF_LINE_TOKEN, END_OF_LINE_TOKEN]

        return tokens
    """


class RWKVChat(RWKVEmbryo):
    def __init__(self, id: str, state_name: str = model_state_name, prompt: str = None):
        super().__init__(id, state_name, prompt)

    def chat(
        self,
        msg: str,
        chatuser: str = user,
        nickname: str = bot,
        show_user_to_model: bool = False,
    ):
        self.ulog.write(f"{chatuser}: {msg}\n")
        temperature: float = TEMPERATURE
        top_p: float = TOP_P

        if "-temp=" in msg:
            temperature = float(msg.split("-temp=")[1].split(" ")[0])
            msg = msg.replace("-temp=" + f"{temperature:g}", "")
            temperature = max(0.2, min(temperature, 5.0))

        if "-top_p=" in msg:
            top_p = float(msg.split("-top_p=")[1].split(" ")[0])
            msg = msg.replace("-top_p=" + f"{top_p:g}", "")
            top_p = max(0.2, min(top_p, 5.0))

        if "+reset" in msg:
            self.reset()
            return " : Done"

        if not show_user_to_model:  # 不向模型展示用户名
            msg = msg.replace(chatuser, user)
        msg = msg.replace(nickname, bot)  # .strip() # 昵称和提示词不一定一致

        with self.process_lock:
            if msg != "+":
                new = f"{chatuser}{separator} {msg}\n\n{nickname}{separator}"
                self.process_tokens(tokenizer.encode(new))

            answer: bytes = b""
            start_index: int = len(self.processed_tokens)
            for i in tqdm.trange(
                MAX_GENERATION_LENGTH,
                desc="Processing future",
                leave=False,
                unit=" tok",
            ):
                self.process_token_penalty()
                token: int = sampling.sample_logits(self.logits, temperature, top_p)

                if token == END_OF_TEXT_TOKEN:
                    break
                else:
                    self.process_token(token)

                answer += tokenizer.decodeBytes([token])
                if b"\n\n" in answer:
                    break

        answer = answer.decode("utf-8").strip()
        if not show_user_to_model:  # 把昵称和用户名换回去
            answer = answer.replace(user, chatuser)
        answer = answer.replace(bot, nickname).strip()

        self.ulog.write(f"{nickname}: {answer}\n")
        self.save_state(self.id)
        return answer


def process_default_state():
    if os.path.isfile(f"data/{model_state_name}/tokens.pkl"):
        prxxx("Default state was processed")
    else:
        RWKVChat(
            id="chat-model", state_name=model_state_name, prompt=default_init_prompt
        )
