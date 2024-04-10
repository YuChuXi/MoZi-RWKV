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
import copy
from rwkv_cpp import rwkv_cpp_shared_library, rwkv_cpp_model
from rwkv_cpp.rwkv_world_tokenizer import RWKV_TOKENIZER

# from tokenizer_util import get_tokenizer
from typing import List, Dict, Optional
from app_util import (
    prxxx,
    check_dir,
    check_file,
    log_call,
    use_async_lock,
    check_dir_async,
    check_file_async,
    run_in_async_thread,
)

from config import (
    MAX_GENERATION_LENGTH,
    TEMPERATURE,
    TOP_P,
    PRESENCE_PENALTY,
    FREQUENCY_PENALTY,
    PRPEAT_PENALTY,
    PENALTY_MITIGATE,
    OBSTINATE,
    END_OF_TEXT_TOKEN,
    THREADS,
    MODEL_NAME,
    MODEL_STATE_NAME,
    TONKEIZER_DICT,
    CHAT_LANGUAGE,
    CHAT_PROMPT_TYPE,
    NICKGENER_PROMPT,
)

library = rwkv_cpp_shared_library.load_rwkv_shared_library()
prxxx(f"System info: {library.rwkv_get_system_info_string()}")
prxxx(f"Loading RWKV model   file: model/{MODEL_NAME}.bin")
model = rwkv_cpp_model.RWKVModel(
    library, f"model/{MODEL_NAME}.bin", thread_count=THREADS
)
check_dir("data")
if check_file(f"data/tokenizer.pkl"):
    prxxx(f"Loading tokenizer   file: data/tokenizer.pkl")
    with open(f"data/tokenizer.pkl", "rb") as f:
        tokenizer: RWKV_TOKENIZER = pickle.load(f)
else:
    prxxx(f"Loading tokenizer   file: {TONKEIZER_DICT}")
    tokenizer: RWKV_TOKENIZER = RWKV_TOKENIZER(TONKEIZER_DICT)
    with open(f"data/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)


# ========================================= Embryo states =========================================


class RWKVState:
    def __init__(self):
        self.logits: np.ndarray = None
        self.state: np.ndarray = None
        self.processed_tokens: List[int] = []
        self.processed_tokens_counts: Dict[int, int] = {}

        self.data = ["logits", "state", "processed_tokens", "processed_tokens_counts"]

    @run_in_async_thread
    def save(self, state_name: str):
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
        np.save(f"data/{state_name}/state.npy", self.state)
        return self

    @run_in_async_thread
    def load(self, state_name: str):
        if not check_file(f"data/{state_name}/tokens.pkl"):
            return None

        with open(f"data/{state_name}/tokens.pkl", "rb") as f:
            data: Dict[str, object] = pickle.load(f)

        self.processed_tokens: List[int] = data["processed_tokens"]
        self.logits: np.ndarray = data["logits"]
        self.processed_tokens_counts: Dict[int, int] = data["processed_tokens_counts"]
        self.state: np.ndarray = np.load(f"data/{state_name}/state.npy")

        return self

    @run_in_async_thread
    def copy(self):
        return copy.deepcopy(self)

    async def mix(self, state, weight: float):
        staot0 = await self.copy()

        staot0.state = staot0.state * (1 - weight) + state.state * weight
        staot0.logits = staot0.logits * (1 - weight) + state.logits * weight

        return staot0

    @run_in_async_thread
    def mix_inplace(self, state, weight: float):
        self.state = self.state * (1 - weight) + state.state * weight
        self.logits = self.logits * (1 - weight) + state.logits * weight

        return self

    async def mix_max(self, state, weight: float):
        staot0 = await self.copy()

        staot0.state = np.maximum(staot0.state, state.state)
        staot0.logits = np.maximum(staot0.logits, state.logits)

        return staot0

    @run_in_async_thread
    def mix_max_inplace(self, state, weight: float):
        self.state = np.maximum(self.state, state.state)
        self.logits = np.maximum(self.logits, state.logits)

        return self


state_cache: Dict[str, RWKVState] = {}


# ============================================ Embryo =============================================


class RWKVEmbryo:
    def __init__(self, id: str, state_name: str = MODEL_STATE_NAME, prompt: str = None):
        prxxx(
            f"Init RWKV   id: {id} | state: {state_name} | prompt: {'None' if prompt is None else prompt.strip().splitlines()[0]}"
        )
        check_dir(f"data/{id}")

        assert len(id) > 0, "ID must not be empty"
        assert not state_name is None and len(state_name) > 0, "State must not be empty"
        assert id != state_name, "ID != State !!!"

        self.id: str = str(id)
        self.prompt: str = prompt
        self.default_state: str = state_name
        self.process_lock = asyncio.Lock()

        self.state = RWKVState()
        self.need_save = False

        self.presence_penalty: float = PRESENCE_PENALTY
        self.frequency_penalty: float = FREQUENCY_PENALTY
        self.repeat_penalty: float = PRPEAT_PENALTY
        self.penalty_mitigate: float = PENALTY_MITIGATE
        self.temperature: float = TEMPERATURE
        self.top_p: float = TOP_P

        self.mlog = open(f"data/{self.id}/model.log", "ab+")

    def __del__(self):
        self.mlog.close()

    @log_call
    async def load_state(
        self, state_name: str, prompt: str = None, reprompt=False, q: bool = False
    ):
        self.mlog.write(f"\n\n : Load[{state_name}]".encode("utf-8"))

        if (prompt is not None) and (
            reprompt
            or (not await check_file_async(f"data/{self.default_state}/tokens.pkl"))
        ):
            prompt_tokens = tokenizer.encode(prompt)
            prxxx(f"Process prompt tokens   length: {len(prompt_tokens)} tok", q=q)
            ltime = time.time()
            await self.process_tokens(prompt_tokens)
            prxxx(f"Processed prompt tokens   used: {int(time.time()-ltime)} s", q=q)
            await self.save_state(self.id, must=True, q=q)
            await self.save_state(self.default_state, must=True, q=q)
            return

        state_names = [self.default_state, MODEL_STATE_NAME]
        if state_name is not None:
            state_names = [state_name] + state_names

        for state_name in state_names:
            await asyncio.sleep(0)
            if (state_name != self.id) and (state_name in state_cache):
                self.state = await state_cache[state_name].copy()
                prxxx(f"Load state from cache   name: {state_name}", q=q)
            else:
                if await self.state.load(state_name) is None:
                    continue
                if state_name != self.id:
                    state_cache[state_name] = await self.state.copy()
                    self.need_save = True
                prxxx(f"Load state   name: {state_name}", q=q)
            break

    @log_call
    async def save_state(self, state_name: str, must: bool = False, q: bool = False):
        if self.need_save or must:
            await self.state.save(state_name)
            prxxx(f"Save state   name: {state_name}", q=q)
            self.need_save = False
        self.mlog.flush()


    @log_call
    async def reset_state(self, q: bool = False):
        await self.load_state(self.default_state, q=q)
        await self.save_state(self.id, must=True, q=q)

    async def init_state(self):
        await self.load_state(self.id, self.prompt)

    @log_call
    async def check_state(self):
        return
        logit = self.logits[self.logits >= 0]
        prxxx("logits", logit[-128:])
        prxxx("pedt", self.state.processed_tokens_counts)
        pppp = list(
            map(
                lambda x: self.repeat_penalty**x,
                self.state.processed_tokens_counts.values(),
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
        self.state.processed_tokens.append(token)
        if token not in self.state.processed_tokens_counts:  # 词频统计
            self.state.processed_tokens_counts[token] = 1
        else:
            self.state.processed_tokens_counts[token] += 1

        for token in self.state.processed_tokens_counts:
            self.state.processed_tokens_counts[token] /= self.penalty_mitigate

    @log_call
    async def process_token_penalty(self, logits: np.ndarray) -> np.ndarray:
        logits[END_OF_TEXT_TOKEN] = -1e9
        for token in self.state.processed_tokens_counts:
            logits[token] -= (
                # 传统惩罚
                self.presence_penalty
                + self.state.processed_tokens_counts[token] * self.frequency_penalty
                # 新惩罚
                + self.repeat_penalty ** self.state.processed_tokens_counts[token]
                - 1
            )
        return logits

    @log_call
    async def process_tokens(self, tokens: List[int]):
        """
        self.logits, self.state = model.eval_sequence(
            tokens, self.state, self.state, self.logits, use_numpy=True)
        self.state.processed_tokens += tokens
        #self.logits[END_OF_LINE_TOKEN] += new_line_logit_bias
        """

        for token in tqdm.tqdm(
            tokens, desc="Processing prompt", leave=False, unit=" tok"
        ):
            await asyncio.sleep(0)
            self.state.logits, self.state.state = model.eval(
                token, self.state.state, self.state.state, self.state.logits
            )
            await self.process_processed_tokens_counts(token)
            self.need_save = True
            await self.check_state()
        self.mlog.write(tokenizer.decodeBytes(tokens))
        return self.state.logits, self.state.state

    @log_call
    async def process_token(self, token: int):
        await asyncio.sleep(0)
        self.state.logits, self.state.state = model.eval(
            token, self.state.state, self.state.state, self.state.logits
        )

        await self.process_processed_tokens_counts(token)
        self.need_save = True
        await self.check_state()
        self.mlog.write(tokenizer.decodeBytes([token]))
        return self.state.logits, self.state.state

    async def gen_future(
        self, max_len: int = MAX_GENERATION_LENGTH, end_of: str = "\n\n"
    ) -> str:
        async with self.process_lock:
            answer: bytes = b""
            end: bytes = end_of.encode("utf-8")
            for i in tqdm.trange(
                max_len,
                desc="Processing future",
                leave=False,
                unit=" tok",
            ):
                await asyncio.sleep(0)
                logits = self.state.logits
                logits = await self.process_token_penalty(logits)
                token: int = sampling.sample_logits(
                    logits, self.temperature, self.top_p
                )
                answer += tokenizer.decodeBytes([token])
                await self.process_token(token)
                if end in answer:
                    break

            self.need_save = True
        return answer.decode("utf-8", errors="ignore").strip()

    async def call(self, api: str, kwargs: Dict[str, object]):
        return await getattr(self, api)(**kwargs)


# ======================================== Chater Embryo ==========================================

prompt_config = f"prompt/{CHAT_LANGUAGE}-{CHAT_PROMPT_TYPE}.json"
prxxx(f"Loading RWKV prompt   config: {prompt_config}")
with open(prompt_config, "r", encoding="utf-8", errors="ignore") as json_file:
    prompt_data = json.load(json_file)
    user, bot, separator, default_init_prompt = (
        prompt_data["user"],
        prompt_data["bot"],
        prompt_data["separator"],
        prompt_data["prompt"],
    )
    if check_file(default_init_prompt):
        with open(default_init_prompt, "r", encoding="utf-8", errors="ignore") as f:
            default_init_prompt = f.read()
assert default_init_prompt != "", "Prompt must not be empty"


class RWKVChaterEmbryo(RWKVEmbryo):
    def __init__(self, id: str, state_name: str = MODEL_STATE_NAME, prompt: str = None):
        super().__init__(id, state_name, prompt)

    async def gen_prompt(
        self,
        message_list: List[List[object]],
        time_limit: float = 1800,
        ctx_limit: int = 1024,
    ):
        """
        [
            [[],[],float],
        #    u  m  t
        ]
        """
        now_time = time.time()
        tokens_list = [
            tokenizer.encode(f"{m[0]}{separator} {m[1]}\n\n")
            for m in message_list
            if now_time - m[2] <= time_limit
        ]
        tokens_list.append(tokenizer.encode(f"{bot}{separator}"))

        prompt = []
        for tl in tokens_list[::-1]:
            len_token = len(tl)
            if len_token <= ctx_limit:
                ctx_limit -= len_token
                prompt = tl + prompt
            else:
                break

        return prompt


# ============================================ Chater =============================================


class RWKVChater(RWKVChaterEmbryo):
    def __init__(self, id: str, state_name: str = MODEL_STATE_NAME, prompt: str = None):
        super().__init__(id, state_name, prompt)

    async def chat(
        self,
        message: str,
        chatuser: str = user,
        nickname: str = bot,
    ):
        if "-temp=" in message:
            temperature = float(message.split("-temp=")[1].split(" ")[0])
            message = message.replace("-temp=" + f"{temperature:g}", "")
            self.temperature = max(0.2, min(temperature, 5.0))

        if "-top_p=" in message:
            top_p = float(message.split("-top_p=")[1].split(" ")[0])
            message = message.replace("-top_p=" + f"{top_p:g}", "")
            self.top_p = max(0.2, min(top_p, 5.0))

        if "+reset" in message:
            await self.reset_state()
            return " : Done"

        message = message.replace(chatuser, user)
        message = message.replace(nickname, bot)  # .strip() # 昵称和提示词不一定一致

        if message != "+":
            new = f"{chatuser}{separator} {message}\n\n{nickname}{separator}"
            await self.process_tokens(tokenizer.encode(new))

        answer = await self.gen_future(end_of="\n\n")
        await self.state.mix_inplace(state_cache[self.default_state], OBSTINATE)
        #await self.state.mix_inplace(state_cache[self.default_state], OBSTINATE)

        answer = answer.replace(user, chatuser)
        answer = answer.replace(bot, nickname).strip()

        return answer


# ========================================= Group Chater ==========================================


class RWKVGroupChater(RWKVChaterEmbryo):
    def __init__(self, id: str, state_name: str = MODEL_STATE_NAME, prompt: str = None):
        super().__init__(id, state_name, prompt)
        self.message_cache: List[List[object]] = []

    async def send_message(self, message: str, chatuser: str = user) -> None:
        if "-temp=" in message:
            temperature = float(message.split("-temp=")[1].split(" ")[0])
            message = message.replace("-temp=" + f"{temperature:g}", "")
            self.temperature = max(0.2, min(temperature, 5.0))

        if "-top_p=" in message:
            top_p = float(message.split("-top_p=")[1].split(" ")[0])
            message = message.replace("-top_p=" + f"{top_p:g}", "")
            self.top_p = max(0.2, min(top_p, 5.0))

        if "+reset" in message:
            await self.reset_state()
            self.message_cache.clear()
            return
    
        self.message_cache.append([chatuser, message, time.time()])
        if len(self.message_cache) > 128:
            self.message_cache = self.message_cache[64]

    async def get_answer(
        self,
        nickname: str = bot,
    ) -> str:
        await self.process_tokens(await self.gen_prompt(self.message_cache))
        self.message_cache.clear()

        answer = await self.gen_future(end_of="\n\n")
        await self.state.mix_inplace(state_cache[self.default_state], OBSTINATE)

        answer = answer.replace(bot, nickname).strip()

        return answer


# ======================================= Nickname Gener ==========================================


class RWKVNicknameGener(RWKVEmbryo):
    def __init__(self):
        super().__init__(
            "-G_RWKVNickNameGener_G", "-S_RWKVNickNameGener_S", NICKGENER_PROMPT
        )
        self.temperature: float = 0.3
        self.top_p: float = 0.1
        self.penalty_mitigate = 0.98
        self.presence_penalty = -1
        self.repeat_penalty = 1
        self.frequency_penalty = 0

    async def gen_nickname(self, name):
        self.state.processed_tokens = []
        self.state.processed_tokens_counts = {}
        new = f"{name}\n"
        await self.process_tokens(tokenizer.encode(new))
        answer = await self.gen_future(max_len=10, end_of="\n\n")

        await self.reset_state(q=True)
        return answer


# ========================================== Other ================================================


async def process_default_state():
    if await check_file_async(f"data/{MODEL_STATE_NAME}/tokens.pkl"):
        prxxx("Default state was processed")
    else:
        await RWKVChater(
            id="chat-model", state_name=MODEL_STATE_NAME, prompt=default_init_prompt
        ).init_state()


"""
print(tokenizer.decode(RWKVChaterEmbryo.gen_prompt(None,[
    ["saefsgrgdr","jgjgjghjghghgjh",time.time()-3600],
    ["hjhjvhvjhb","ftjhvjhjhjhjdsr",time.time()-2400],
    ["guiyutftfd","pohhnkftfgheshj",time.time()-1200],
    ["bnmvnbmcgf","dtrfttdtytyrrr3",time.time()],
    ["uigyfyffrt","jkhfhhgttdhdrrr",time.time()],
    
],time_limit=3600,ctx_limit=1)))
# """
