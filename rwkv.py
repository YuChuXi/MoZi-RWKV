# -*- coding: utf-8 -*-
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
import asyncio
from rwkv_cpp import rwkv_cpp_shared_library, rwkv_cpp_model
from rwkv_cpp.rwkv_world_tokenizer import RWKV_TOKENIZER

# from tokenizer_util import get_tokenizer
from typing import List, Dict, Optional
from app_util import prxxx, check_dir, check_file, log_call, use_async_lock, check_dir_async,check_file_async

# ======================================== Script settings ========================================

MAX_GENERATION_LENGTH: int = 128

# Sampling temperature. It could be a good idea to increase temperature when top_p is low.
TEMPERATURE: float = 1.0
# For better Q&A accuracy and less diversity, reduce top_p (to 0.5, 0.2, 0.1 etc.)
TOP_P: float = 0.5
# Penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
PRESENCE_PENALTY: float = 0.7
# Penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.

FREQUENCY_PENALTY: float = 1.0
# When the model repeats several words, the penalty will increase sharply and pull the model back, set it to 1.0-1.2 is a good idea.
PRPEAT_PENALTY: float = 1.05

# a?
PENALTY_MITIGATE: float = 1.02

# END_OF_LINE_TOKEN: int = 187
# DOUBLE_END_OF_LINE_TOKEN: int = 535
END_OF_TEXT_TOKEN: int = 0

THREADS: int = 8


np.random.seed(int(time.time() * 1e6 % 2**30))

model_name = "RWKV-5-Qun-1B5-Q4_0"
model_name = "RWKV-5-World-3B-Q5_0-v2"
model_name = "RWKV-5-World-1B5-Q5_1-v2"
model_name = "RWKV-5-World-7B-Q5_1-v2"

model_path = f"model/{model_name}.bin"

model_state_name = "default.state"
model_state_path = f"data/{model_state_name}.pkl"

tokenizer_dict = "rwkv_cpp/rwkv_vocab_v20230424.txt"


library = rwkv_cpp_shared_library.load_rwkv_shared_library()
prxxx(f"System info: {library.rwkv_get_system_info_string()}")
# '''
prxxx(f"Loading RWKV model: {model_path}")
model = rwkv_cpp_model.RWKVModel(library, model_path, thread_count=THREADS)
# '''
check_dir("data")
if check_file(f"data/tokenizer.pkl"):
    prxxx(f"Loading tokenizer: data/tokenizer.pkl")
    with open(f"data/tokenizer.pkl", "rb") as f:
        tokenizer: RWKV_TOKENIZER = pickle.load(f)
else:
    prxxx(f"Loading tokenizer: {tokenizer_dict}")
    tokenizer: RWKV_TOKENIZER = RWKV_TOKENIZER(tokenizer_dict)
    with open(f"data/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

# ======================================== Embryo settings ========================================

state_cache: Dict[str, Dict[str, object]] = {}

# =================================================================================================


class RWKVEmbryo:
    def __init__(self, id: str, state_name: str = model_state_name, prompt: str = None):
        prxxx(
            f"Init RWKV id: {id} | state: {state_name} | prompt: {'None' if prompt is None else prompt.strip().splitlines()[0]}"
        )
        check_dir(f"data/{id}")

        self.id: str = str(id)
        self.default_state: str = state_name
        self.process_lock: Lock = Lock()

        self.logits: np.ndarray = None
        self.state: np.ndarray = None
        self.processed_tokens: List[int] = []
        self.processed_tokens_counts: Dict[int, int] = {}

        self.presence_penalty: float = PRESENCE_PENALTY
        self.frequency_penalty: float = FREQUENCY_PENALTY
        self.repeat_penalty: float = PRPEAT_PENALTY
        self.penalty_mitigate: float = PENALTY_MITIGATE
        self.temperature: float = TEMPERATURE
        self.top_p: float = TOP_P

        self.mlog = open(f"data/{self.id}/model.log", "ab+")
        self.ulog = open(f"data/{self.id}/user.log", "a+", encoding="utf-8")
        self.load_state(self.id, prompt)

    def __del__(self):
        self.mlog.close()
        self.ulog.close()

    @use_async_lock
    @log_call
    async def load_state(
        self, state_name: str, prompt: str = None, reprompt=False, q: bool = False
    ):
        self.mlog.write(f" : Load[{state_name}]\n\n".encode("utf-8"))

        if (prompt is not None) and (
            reprompt or (not await check_file_async(f"data/{self.default_state}/tokens.pkl"))
        ):
            prompt_tokens = tokenizer.encode(prompt)
            prxxx(f"Process prompt tokens, length: {len(prompt_tokens)} tok", q=q)
            ltime = time.time()
            await self.process_tokens(prompt_tokens)
            prxxx(f"Processed prompt tokens, used: {int(time.time()-ltime)} s", q=q)
            await self.save_state(self.id, q=q)
            await self.save_state(self.default_state, q=q)
            return

        state_names = [self.default_state, model_state_name]
        if state_name is not None:
            state_names = [state_name] + state_names

        for state_name in state_names:
            await asyncio.sleep(0)
            if (state_name != self.id) and (state_name in state_cache):
                prxxx(f"Load state from cache: {state_name}", q=q)
                data = state_cache[state_name].copy()
            else:
                if not await check_file_async(f"data/{state_name}/tokens.pkl"):
                    continue
                prxxx(f"Load state: {state_name}", q=q)
                with open(f"data/{state_name}/tokens.pkl", "rb") as f:
                    data: Dict[str, object] = pickle.load(f)
                if state_name != self.id:
                    state_cache[state_name] = data.copy()

            self.processed_tokens: List[int] = data["processed_tokens"]
            self.logits: np.ndarray = data["logits"]
            self.processed_tokens_counts: Dict[int, int] = data[
                "processed_tokens_counts"
            ]
            self.state: np.ndarray = np.load(f"data/{state_name}/state.npy")
            break

    @use_async_lock
    @log_call
    async def save_state(self, state_name: str, q: bool = False):
        prxxx(f"Save state: {state_name}", q=q)
        await check_dir_async(f"data/{state_name}")
        with open(f"data/{state_name}/tokens.pkl", "wb") as f:
            pickle.dump(
                {
                    "processed_tokens": self.processed_tokens,
                    "logits": self.logits,
                    "processed_tokens_counts": self.processed_tokens_counts,
                },
                f,
            )
        np.save(f"data/{state_name}/state.npy", self.state)

    @use_async_lock
    @log_call
    async def reset(self, quiet: bool = False, q: bool = False):
        await self.load_state(self.default_state, q=q)
        self.ulog.write(" : Reset")
        await self.save_state(self.id, q=q)

    @log_call
    async def check_state(self):
        return
        logit = self.logits[self.logits >= 0]
        prxxx("logits", logit[-128:])
        prxxx("pedt", self.processed_tokens_counts)
        pppp = list(
            map(
                lambda x: self.repeat_penalty**x,
                self.processed_tokens_counts.values(),
            )
        )
        pppp.sort()
        prxxx("pppp", pppp)
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

    @log_call
    async def process_processed_tokens_counts(self, token: int):
        self.processed_tokens.append(token)
        if token not in self.processed_tokens_counts:  # 词频统计
            self.processed_tokens_counts[token] = 1
        else:
            self.processed_tokens_counts[token] += 1

        for token in self.processed_tokens_counts:
            self.processed_tokens_counts[token] /= self.penalty_mitigate

    @log_call
    async def process_token_penalty(self):
        self.logits[END_OF_TEXT_TOKEN] = -1e9
        for token in self.processed_tokens_counts:
            self.logits[token] -= (
                # 传统惩罚
                self.presence_penalty
                + self.processed_tokens_counts[token] * self.frequency_penalty
                # 新惩罚
                + self.repeat_penalty ** self.processed_tokens_counts[token]
                - 1
            )

    @use_async_lock
    @log_call
    async def process_tokens(
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
            await asyncio.sleep(0)
            self.logits, self.state = model.eval(
                    token, self.state, self.state, self.logits
                )
            await self.process_processed_tokens_counts(token)
            await self.check_state()
        # self.logits[END_OF_LINE_TOKEN] += new_line_logit_bias
        self.mlog.write(tokenizer.decodeBytes(tokens))
        return self.logits, self.state

    @use_async_lock
    @log_call
    async def process_token(self, token: int, new_line_logit_bias: float = 0.0) -> None:
        await asyncio.sleep(0)
        self.logits, self.state = model.eval(
                token, self.state, self.state, self.logits
            )

        await self.process_processed_tokens_counts(token)
        # self.logits[END_OF_LINE_TOKEN] += new_line_logit_bias
        await self.check_state()
        self.mlog.write(tokenizer.decodeBytes([token]))
        return self.logits, self.state

    """
    # Model only saw '\n\n' as [187, 187] before, but the tokenizer outputs [535] for it at the end.
    # See https://github.com/Blink/pulEmbryo:/110/files


    def split_last_end_of_line(self, tokens: List[int]) -> List[int]:
        if len(tokens) > 0 and tokens[-1] == DOUBLE_END_OF_LINE_TOKEN:
            tokens = tokens[:-1] + [END_OF_LINE_TOKEN, END_OF_LINE_TOKEN]

        return tokens
    """


# ========================================= Chat settings =========================================

# English, Chinese, Japanese
language: str = "Chinese"
# QA: Question and Answer prompt to talk to an AI assistant.
# Chat: chat prompt (need a large model for adequate quality, 7B+).
prompt_type: str = "Chat-MoZi-N"


prompt_config = f"prompt/{language}-{prompt_type}.json"
prxxx(f"Loading RWKV prompt config: {prompt_config}")
with open(prompt_config, "r", encoding="utf-8") as json_file:
    prompt_data = json.load(json_file)
    user, bot, separator, default_init_prompt = (
        prompt_data["user"],
        prompt_data["bot"],
        prompt_data["separator"],
        prompt_data["prompt"],
    )
    if check_file(default_init_prompt):
        with open(default_init_prompt, "rb") as f:
            default_init_prompt = f.read().decode("utf-8")
assert default_init_prompt != "", "Prompt must not be empty"


# =================================================================================================
class RWKVChaterEmbryo(RWKVEmbryo):
    def __init__(self, id: str, state_name: str = model_state_name, prompt: str = None):
        super().__init__(id, state_name, prompt)

    async def gen_prompt(
        self,
        message_list: List[List[object]],
        time_limit: float = 3600,
        ctx_limit: int = 256,
    ):
        """
        [
            [[],[],float],
        #    u  m  t
        ]
        """
        now_time = time.time()
        tokens_list = [tokenizer.encode(f"{m[0]}{separator} {m[1]}\n\n") for m in message_list if now_time - m[2] <= time_limit]
        tokens_list.append(tokenizer.encode(f"{bot}{separator} "))
        for i in range(len(tokens_list),0,-1):
            len_token = len(tokens_list[i-1])
            if len_token <= ctx_limit:
                ctx_limit -= len_token
            else:
                break
        prompt = []
        for l in tokens_list[i-1:]:
            prompt += l
        return prompt

    @use_async_lock
    async def gen_answer(self, end_of: str = "\n\n") -> str:
        answer: bytes = b""
        end: bytes = end_of.encode("utf-8")
        for i in tqdm.trange(
            MAX_GENERATION_LENGTH,
            desc="Processing future",
            leave=False,
            unit=" tok",
        ):
            await asyncio.sleep(0)
            await self.process_token_penalty()
            token: int = sampling.sample_logits(
                self.logits, self.temperature, self.top_p
            )
            await self.process_token(token)
            answer += tokenizer.decodeBytes([token])
            if end in answer:
                break
        return answer.decode("utf-8").strip()


class RWKVChater(RWKVChaterEmbryo):
    def __init__(self, id: str, state_name: str = model_state_name, prompt: str = None):
        super().__init__(id, state_name, prompt)

    @use_async_lock
    async def chat(
        self,
        message: str,
        chatuser: str = user,
        nickname: str = bot,
    ):
        self.ulog.write(f"{chatuser}: {message}\n")

        if "-temp=" in message:
            temperature = float(message.split("-temp=")[1].split(" ")[0])
            message = message.replace("-temp=" + f"{temperature:g}", "")
            self.temperature = max(0.2, min(temperature, 5.0))

        if "-top_p=" in message:
            top_p = float(message.split("-top_p=")[1].split(" ")[0])
            message = message.replace("-top_p=" + f"{top_p:g}", "")
            self.top_p = max(0.2, min(top_p, 5.0))

        if "+reset" in message:
            self.reset()
            return " : Done"

        message = message.replace(chatuser, user)
        message = message.replace(nickname, bot)  # .strip() # 昵称和提示词不一定一致

        with self.process_lock:
            if message != "+":
                new = f"{chatuser}{separator} {message}\n\n{nickname}{separator}"
                await self.process_tokens(tokenizer.encode(new))
            answer = await self.gen_answer(end_of="\n\n")

        answer = answer.replace(user, chatuser)
        answer = answer.replace(bot, nickname).strip()

        self.ulog.write(f"{nickname}: {answer}\n")
        # self.save_state(self.id, q=True)
        return answer


class RWKVGroupChater(RWKVChaterEmbryo):
    def __init__(self, id: str, state_name: str = model_state_name, prompt: str = None):
        super().__init__(id, state_name, prompt)
        self.message_cache: List[List[object]] = []

    @use_async_lock
    async def send_message(self, message: str, chatuser: str = user) -> None:
        self.message_cache.append(
            [chatuser, message, time.time()]
        )

    @use_async_lock
    async def get_answer(
        self,
        nickname: str = bot,
    ) -> str:
        message = message.replace(nickname, bot)  # .strip() # 昵称和提示词不一定一致

        with self.process_lock:
            if message != "+":
                new = await self.gen_prompt(self.message_cache)
                await self.process_tokens(tokenizer.encode(new))
            answer = await self.gen_answer(end_of="\n\n")
        answer = answer.replace(bot, nickname).strip()

        self.ulog.write(f"{nickname}: {answer}\n")
        # self.save_state(self.id, q=True)
        return answer


# ======================================== Gener settings =========================================
prompt = """注: 
以下是一张用户名与称呼的对照表，称呼是用户名中最具有特色的部分, 长度在五个字以内. 

用户名: 玉子是个废物喵
称呼: 玉子

用户名: 沐沐
称呼: 沐沐

用户名: 咦我的名字呢？
称呼: 没名字

用户名: YuChuXi
称呼: YuChuXi

用户名: 墨子不是猫
称呼: 墨子

用户名: 不想加班的朋朋
称呼: 朋朋

用户名: 只有鱼骨头吃的喵
称呼: 鱼骨头喵

用户名: 猫猫博士凌枫
称呼: 猫猫博士


"""


# =================================================================================================
class RWKVNicknameGener(RWKVEmbryo):
    def __init__(self):
        super().__init__("-G_RWKVNickNameGener_G", "-S_RWKVNickNameGener_S", prompt)
        self.temperature: float = TEMPERATURE
        self.top_p: float = TOP_P

    async   def gen_nickname(self, name):
        self.ulog.write(f"用户名: {name}\n称呼: ")

        with self.process_lock:
            new = f"用户名: {name}\n称呼: "
            await self.process_tokens(tokenizer.encode(new))

            answer: bytes = b""
            for i in tqdm.trange(
                MAX_GENERATION_LENGTH,
                desc="Processing future",
                leave=False,
                unit=" tok",
            ):
                await asyncio.sleep(0)
                await self.process_token_penalty()
                token: int = sampling.sample_logits(
                    self.logits, self.temperature, self.top_p
                )

                if token == END_OF_TEXT_TOKEN:
                    break
                else:
                    await self.process_token(token)

                answer += tokenizer.decodeBytes([token])
                if b"\n\n" in answer:
                    break

        answer = answer.decode("utf-8").strip()
        await self.reset(q=True)
        return answer


def process_default_state():
    if check_file(f"data/{model_state_name}/tokens.pkl"):
        prxxx("Default state was processed")
    else:
        RWKVChater(
            id="chat-model", state_name=model_state_name, prompt=default_init_prompt
        )

'''
print(tokenizer.decode(RWKVChaterEmbryo.gen_prompt(None,[
    ["saefsgrgdr","jgjgjghjghghgjh",time.time()-3600],
    ["hjhjvhvjhb","ftjhvjhjhjhjdsr",time.time()-2400],
    ["guiyutftfd","pohhnkftfgheshj",time.time()-1200],
    ["bnmvnbmcgf","dtrfttdtytyrrr3",time.time()],
    ["uigyfyffrt","jkhfhhgttdhdrrr",time.time()],
    
],time_limit=3600,ctx_limit=1)))
# '''