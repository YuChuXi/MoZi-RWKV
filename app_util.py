import time, re, random, os, sys
from typing import Callable
import asyncio

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


def log_call(func):
    def nfunc(*args, **kwargs):
        prxxx(f"Call {func.__name__}", from_debug=True)
        return func(*args, **kwargs)
    return nfunc

def use_async_lock(func):
    lock = asyncio.locks.Lock()
    async def nfunc(*args, **kwargs):
        async with lock:
            return await func(*args, **kwargs)
    return nfunc

def run_in_async_thread(func):
    if sys.version_info.micro < 9:
        return func
    async def nfunc(*args, **kwargs):
        thread = asyncio.to_threads(func, *args, **kwargs)
        return await thread
    return nfunc

symbols = "[!@#$%^&*+[\]{};:/<>?\|`~]"

def clean_symbols(s):
    return "".join([c for c in s if c not in symbols]) 


def gen_echo():
    return "%4.x" % random.randint(0, 65535)


def check_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def check_file(path):
    return os.path.isfile(path)


@run_in_async_thread
def check_dir_async(path):
    if not os.path.isdir(path):
        os.makedirs(path)

@run_in_async_thread
def check_file_async(path):
    return os.path.isfile(path)
