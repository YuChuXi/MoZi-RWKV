import numpy as np
import time
from typing import List

# ======================================== Script settings ========================================

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
# Mitigating penalties after a certain length of context
PENALTY_MITIGATE: float = 1.02
# How engaged a model is with prompt, which could be used to mitigate Alzheimer's disease in small models
OBSTINATE: float = 0.1

MAX_GENERATION_LENGTH: int = 128
END_OF_TEXT_TOKEN: int = 0

THREADS: int = 3

MODEL_NAME: str = "RWKV-5-Qun-1B5-Q4_0"
MODEL_NAME: str = "RWKV-5-World-3B-Q5_0-v2"
MODEL_NAME: str = "RWKV-5-World-7B-Q5_1-v2"
MODEL_NAME: str = "RWKV-6-World-1B6-Q5_1-v2v1"
MODEL_NAME: str = "RWKV-5-World-1B5-Q5_1-v2"

MODEL_STATE_NAME: str = "default.state"

TONKEIZER_DICT: str = "rwkv_cpp/rwkv_vocab_v20230424.txt"

np.random.seed(int(time.time() * 1e6 % 2**30))


# ========================================= App settings ==========================================

APP_BIND: List[str] = ["0.0.0.0:8088", "[::]:8089"]
APP_AUTOSAVE_TIME: int = 600

APP_TEST_MESSAGE: str = """告诉我关于你的一切。"""


# ========================================= Chat settings =========================================

# English, Chinese, Japanese
CHAT_LANGUAGE: str = "Chinese"
# QA: Question and Answer prompt to talk to an AI assistant.
# Chat: chat prompt (need a large model for adequate quality, 7B+).
CHAT_PROMPT_TYPE: str = "Chat-MoZi-N"


# ======================================== Gener settings =========================================

NICKGENER_PROMPT: str = """注: 
以下是一张用户名与称呼的对照表，称呼是用户名中最具有特色的部分, 且尽可能短. 

玉子是个废物喵
玉子

沐沐
沐沐

墨子不是猫
墨子

不想加班的朋朋
朋朋

只有鱼骨头吃的喵
鱼骨头喵

猫猫博士凌枫
猫猫博士


"""
