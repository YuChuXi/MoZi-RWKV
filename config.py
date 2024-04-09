import numpy as np
import time


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
#
OBSTINATE: float = 0.1

# END_OF_LINE_TOKEN: int = 187
# DOUBLE_END_OF_LINE_TOKEN: int = 535
END_OF_TEXT_TOKEN: int = 0

THREADS: int = 3


np.random.seed(int(time.time() * 1e6 % 2**30))

model_name = "RWKV-5-Qun-1B5-Q4_0"
model_name = "RWKV-5-World-3B-Q5_0-v2"
model_name = "RWKV-5-World-7B-Q5_1-v2"
model_name = "RWKV-6-World-1B6-Q5_1-v2v1"
model_name = "RWKV-5-World-1B5-Q5_1-v2"

model_path = f"model/{model_name}.bin"

model_state_name = "default.state"
model_state_path = f"data/{model_state_name}.pkl"

tokenizer_dict = "rwkv_cpp/rwkv_vocab_v20230424.txt"
