import time, re, random, os
from typing import Callable

def prxxx(*args, q: bool = False, from_debug=False, **kwargs):
    if q:
        return
        pass
    if from_debug:
        return
        pass
    print(
        time.strftime(
            "RWKV [\033[33m%Y-%m-%d %H:%M:%S\033[0m] \033[0m", time.localtime()
        ),
        *args,
        "\033[0m",
        **kwargs,
    )


pattern = re.compile("[!@#$%^&*+[\]{};:/<>?\|`~]")


def clean_symbols(s):
    return re.sub(pattern, "", s)


def gen_echo():
    return "%4.x" % random.randint(0, 65535)


def check_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def check_file(path):
    return os.path.isfile(path)


def log_call(func):
    def nfunc(*args, **kwargs):
        prxxx(f"Call {func.__name__}", from_debug=True)
        return func(*args, **kwargs)
    return nfunc