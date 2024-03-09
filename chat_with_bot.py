# Provides terminal-based chat interface for RWKV model.
# Usage: python chat_with_bot.py C:\rwkv.cpp-169M.bin
# Prompts and code adapted from https://github.com/BlinkDL/ChatRWKV/blob/9ca4cdba90efaee25cfec21a0bae72cbd48d8acd/chat.py

import os
import pickle
import json
import tqdm
import time
import numpy
import sampling

from rwkv_cpp import rwkv_cpp_shared_library, rwkv_cpp_model
from tokenizer_util import get_tokenizer
from typing import List, Dict, Optional

import torch
# ======================================== Script settings ========================================

# English, Chinese, Japanese
LANGUAGE: str = 'Chinese'
# QA: Question and Answer prompt to talk to an AI assistant.
# Chat: chat prompt (need a large model for adequate quality, 7B+).
PROMPT_TYPE: str = "Chat-MoZi-2"

MAX_GENERATION_LENGTH: int = 250

# Sampling temperature. It could be a good idea to increase temperature when top_p is low.
TEMPERATURE: float = 0.4
# For better Q&A accuracy and less diversity, reduce top_p (to 0.5, 0.2, 0.1 etc.)
TOP_P: float = 0.4
# Penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
PRESENCE_PENALTY: float = 2.0
# Penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
FREQUENCY_PENALTY: float = 5.0

END_OF_LINE_TOKEN: int = 187
DOUBLE_END_OF_LINE_TOKEN: int = 535
END_OF_TEXT_TOKEN: int = 0

torch.set_num_threads(10)

# =================================================================================================

with open(f"prompt/{LANGUAGE}-{PROMPT_TYPE}.json", "r", encoding="utf8") as json_file:
    prompt_data = json.load(json_file)

    user, bot, separator, init_prompt, nickname = prompt_data["user"], prompt_data[
        "bot"], prompt_data["separator"], prompt_data["prompt"], prompt_data["nickname"]
    if os.path.isfile(init_prompt):
        with open(init_prompt, "rb") as f:
            init_prompt = f.read().decode("utf-8")

assert init_prompt != '', 'Prompt must not be empty'

library = rwkv_cpp_shared_library.load_rwkv_shared_library()
print(f'System info: {library.rwkv_get_system_info_string()}')

print('Loading RWKV model')

model_name = "RWKV-5-World-3B-Q5_0-v2"
# model_name = "RWKV-5-World-1.5B-Q5_0-v2"
model_token_name = "default.state"

model_path = "model/" + model_name + ".bin"
model_token_path = "data/" + model_token_name + ".pkl"

model = rwkv_cpp_model.RWKVModel(library, model_path)

tokenizer_decode, tokenizer_encode = get_tokenizer(
    "world" if "orld" in model_name else "auto", model.n_vocab)

if os.path.isfile(model_token_path):  # load
    print(f"Prompt tokens will use {model_token_path}, load...", end="")
    with open(model_token_path, "rb") as f:
        data = pickle.load(f)
        default_processed_tokens = data["processed_tokens"]
        default_logits = data["logits"]
        default_state = data["state"]
        print("Done.")

else:  # creat
    print(
        f"Prompt tokens will? {model_token_path}, creat...", end="", flush=True)
    prompt_tokens = tokenizer_encode(init_prompt)
    default_logits, default_state = None, None
    print("processing prompt tokens...", end="", flush=True)
    ltime = time.time()
    default_logits, default_state = model.eval_sequence_in_chunks(
        prompt_tokens, default_state, default_state, default_logits, chunk_size=12, use_numpy=True)
    print(f"{time.time()-ltime}s, saving...", end="", flush=True)
    with open(model_token_path, "wb") as f:
        print("Save token...", end="", flush=True)
        pickle.dump({"processed_tokens": [],
                     "logits": default_logits,
                     "state": default_state,
                     }, f)
        print("Done.")

print(
    f"Logits {default_logits}\n\tMin: {float(default_logits.min())}\tMax: {float(default_logits.max())}\tSize: {default_logits.size}")
print(
    f"State  {default_state}\n\tMix: {float(default_state.min())}\tMax: {float(default_state.max())}\tSize: {default_state.size}")

print("RWKV!!!\n")

# =================================================================================================


class ChatRWKV():
    def __init__(self, id, state_name=model_token_name,auto_penalty = True):
        print(f"Init RWKV id:{id}, ", end="", flush=True)
        self.id = str(id)
        self.default_state = state_name
        self.mlog = open(f"data/history/{self.id}.mlog", "ab+")
        self.plog = open(f"data/history/{self.id}.plog", "a+")
        self.presence_penalty = PRESENCE_PENALTY
        self.frequency_penalty = FREQUENCY_PENALTY
        self.load_state(self.id)
        print("Done.")

    def load_state(self, state_name):
        print(state_name)
        if os.path.isfile(f"data/{state_name}.pkl"):
            print(f"State will use data/{state_name}.pkl, load...", end="", flush=True)
            with open(f"data/{state_name}.pkl", "rb") as f:
                data = pickle.load(f)
                self.processed_tokens: list[int] = data["processed_tokens"]
                self.logits: Optional[torch.Tensor] = data["logits"]
                self.state: Optional[torch.Tensor] = data["state"]
        else:
            self.load_state(self.default_state)

        self.mlog.write(f" : Load[{state_name}]\n\n".encode("utf-8"))
        self.mlog.flush()

    def save_state(self, state_name):
        with open(f"data/{state_name}.pkl", "wb") as f:
            pickle.dump({"processed_tokens": self.processed_tokens,
                         "logits": self.logits,
                         "state": self.state}, f)
        self.mlog.flush()
        self.plog.flush()

    def check_state(self):
        return
        l = self.logits#.numpy()
        s = self.state#.numpy()
        s_var = s.var()
        print("*  logits:\tmean\t%.2f\tvar\t%.2f\tmax\t%.2f\tmin %.2f"%(l.mean(),l.var(),l.max(),l.min()))
        print("*  state:\tmean\t%.2f\tvar\t%.2f\tmax\t%.2f\tmin %.2f"%(s.mean(),s_var,s.max(),s.min()))
        self.presence_penalty = s_var/72
        self.frequency_penalty = s_var/36

    def process_tokens(self, tokens: List[int], new_line_logit_bias: float = 0.0) -> None:
        self.logits, self.state = model.eval_sequence(
            tokens, self.state, self.state, self.logits, use_numpy=False)
        self.processed_tokens += tokens
        self.logits[END_OF_LINE_TOKEN] += new_line_logit_bias

        self.check_state()
        self.mlog.write(tokenizer_decode(tokens).encode("utf-8"))

    def process_token(self, token: int, new_line_logit_bias: float = 0.0) -> None:
        self.logits, self.state = model.eval(
            token, self.state, self.state, self.logits, use_numpy=False)
        self.processed_tokens.append(token)
        self.logits[END_OF_LINE_TOKEN] += new_line_logit_bias

        self.check_state()
        self.mlog.write(tokenizer_decode([token]).encode("utf-8"))

    # Model only saw '\n\n' as [187, 187] before, but the tokenizer outputs [535] for it at the end.
    # See https://github.com/BlinkDL/ChatRWKV/pull/110/files

    def split_last_end_of_line(self, tokens: List[int]) -> List[int]:
        if len(tokens) > 0 and tokens[-1] == DOUBLE_END_OF_LINE_TOKEN:
            tokens = tokens[:-1] + [END_OF_LINE_TOKEN, END_OF_LINE_TOKEN]

        return tokens

    # =================================================================================================

    def chat(self, msg, chatuser='你', nickname=nickname):
        self.plog.write(f"{user}: {msg}\n")
        temperature: float = TEMPERATURE
        top_p: float = TOP_P

        if '-temp=' in msg:
            temperature = float(msg.split('-temp=')[1].split(' ')[0])

            msg = msg.replace('-temp='+f'{temperature:g}', '')

            if temperature <= 0.2:
                temperature = 0.2

            if temperature >= 5:
                temperature = 5

        if '-top_p=' in msg:
            top_p = float(msg.split('-top_p=')[1].split(' ')[0])

            msg = msg.replace('-top_p='+f'{top_p:g}', '')

            if top_p <= 0:
                top_p = 0

        msg = msg.replace(chatuser, user).replace(nickname, separator).strip()

        # + reset --> reset chat
        if '+reset' in msg:
            self.load_state(self.default_state)
            self.plog.write(" : Reset")
            self.save_state(self.id)
            print(f'{bot}{separator} Chat reset.\n')
            return " : Done"

        # + --> alternate chat reply
        if msg.lower() != '+':
            new = f'{user}{separator} {msg}\n\n{bot}{separator}'
            self.process_tokens(tokenizer_encode(
                new), new_line_logit_bias=-999999999)
            # Print bot response
            print(f' : {bot}{separator}', end='')

        start_index: int = len(self.processed_tokens)
        accumulated_tokens: List[int] = []
        token_counts: Dict[int, int] = {}
        answer: List[str] = []

        for i in tqdm.trange(MAX_GENERATION_LENGTH, desc="Processing tokens", leave=False, unit=" tok"):
            for n in token_counts:
                self.logits[n] -= self.presence_penalty + token_counts[n] * self.frequency_penalty

            token: int = sampling.sample_logits(
                self.logits, temperature, top_p)

            if token == END_OF_TEXT_TOKEN:
                answer.append("\n")
                break

            if token not in token_counts:
                token_counts[token] = 1
            else:
                token_counts[token] += 1

            self.process_token(token)

            # Avoid UTF-8 display issues
            accumulated_tokens += [token]

            decoded: str = tokenizer_decode(accumulated_tokens)

            if '\uFFFD' not in decoded:
                answer.append(decoded)

                accumulated_tokens = []

            if '\n\n' in tokenizer_decode(self.processed_tokens[start_index:]):
                break

            if i == MAX_GENERATION_LENGTH - 1:
                answer.append("\n")

        answer = "".join(answer).replace(
            user, chatuser).replace(separator, nickname).strip()
        self.plog.write(f"{nickname}: {answer}\n")
        self.save_state(self.id)
        return answer
