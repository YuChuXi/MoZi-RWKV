import time, re, random, os

def prxxx(*args, **kwargs):
    print(
        *([time.strftime("RWKV [\033[33m%Y-%m-%d %H:%M:%S\033[0m] \033[0m", time.localtime())] + list(args)) +["\033[0m"],
        **kwargs,
    )

pattern = re.compile("[!@#$%^&*+[\]{};:/<>?\|`~]")

def clean_symbols(s):
    return re.sub(pattern, "", s)

def gen_echo():
    return "%4.x"%random.randint(0,65535)

def check_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)