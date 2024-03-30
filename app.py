# -*- coding: utf-8 -*-
import time, random, re, sys, os, signal
import uvicorn
import json
import asyncio
from rwkv import RWKVChater, RWKVNicknameGener, process_default_state
from app_util import prxxx, gen_echo, clean_symbols
from typing import Dict

HOST, PORT = ("0.0.0.0", 8088)

test_msg = """告诉我关于你的一切。"""

with open("help.min.html", "r") as f:
    help = f.read()

random.seed(time.time())
chaters: Dict[str, RWKVChater] = {}
process_default_state()

nicknameGener = RWKVNicknameGener()

def restart():
    python = sys.executable
    prxxx("### Restart ! ###")
    os.execl(python, python, *sys.argv)


def stop(signal=None, frame=None):
    prxxx("### STOP ! ###")
    sys.exit(0)

async def save_chaters_state():
    for id in chaters:
        await chaters[id].save_state(id)



async def R_chat(kwargs:Dict[str,object]):
    req_msg: str = kwargs.get("message", default="")
    id: str = clean_symbols(kwargs.get("id", default="-b2bi0JgEhJru87HTcRjh9vdT"))
    user: str = kwargs.get("user", default="木子")
    nickname: str = kwargs.get("nickname", default="墨子")
    multiuser: bool = kwargs.get("multiuser", default=True)
    state: str = kwargs.get("state", default=None)
    # req_msg = req_msg if len(req_msg) <= 256 else req_msg[:256]

    echo = gen_echo()
    prxxx()
    if not id in chaters:
        chaters[id] = RWKVChater(id, state_name=state)
        await chaters[id].init_state()

    prxxx(f" #    Chat id: {id} | user: {user} | echo: {echo}")
    prxxx(f" #    -->[{req_msg}]-{echo}")
    bak_msg = await chaters[id].chat(
        msg=req_msg, chatuser=user, nickname=nickname, show_user_to_model=multiuser
    )
    prxxx(f" #  {echo}-[{bak_msg}]<--")

    # 如果接受到的内容为空，则给出相应的回复
    if bak_msg.isspace() or len(bak_msg) == 0:
        bak_msg = "喵喵喵？"
    return json.dumps({"message": bak_msg, "state": "ok"})



async def R_nickname(kwargs):
    echo = gen_echo()
    name = kwargs["name"]
    prxxx()
    prxxx(f" #    GenNickname echo: {echo}")
    prxxx(f" #    -->[{name}]-{echo}")
    nickname = await nicknameGener.gen_nickname(name)
    prxxx(f" #  {echo}-[{nickname}]<--")

    # 如果接受到的内容为空，则给出相应的回复
    if nickname.isspace() or len(nickname) == 0 or nickname == "None":
        nickname = name
    return json.dumps({"nickname": nickname, "state": "ok"})


async def R_cleanstate(kwargs):
    try:
        id: str = kwargs.get("id")
        if not id in chaters:
            chaters[id] = RWKVChater(id)
            await chaters[id].init_state()
        await chaters[id].reset()
        return json.dumps({"state": "ok"})
    except:
        return """
NM
    """


async def R_restart(kwargs):
    if kwargs.get("passwd_gkd") == "ihAVEcODE":
        await save_chaters_state()
        restart()
    return json.dumps({"state": "fuck you!"})


async def R_stop(kwargs):
    if kwargs.get("passwd_gkd") == "ihAVEcODE":
        await save_chaters_state()
        stop()
    return json.dumps({"state": "fuck you!"})



async def R_index():
    return help


async def test():
    chaters["init"] = RWKVChater("init")
    await chaters["init"].init_state()

    prxxx(f"State size: {chaters['init'].state.size}")
    prxxx(f"State shape: {chaters['init'].state.shape}")
    await chaters["init"].reset()
    echo = gen_echo()
    prxxx(f" #    Test id: test | user: 测试者 | echo:{echo}")
    prxxx(f" #    -->[{test_msg}]-{echo}")
    prxxx(f" #  {echo}-[{chaters['init'].chat(test_msg, chatuser = '测试者')}]<--")



async def app(scope, receive, send):
    raise Exception()

async def main():
    signal.signal(signal.SIGINT, stop)
    await nicknameGener.init_state()
    test()
    prxxx()
    prxxx(" *#*   RWKV！高性能ですから!   *#*")
    prxxx()
    prxxx("Web api server start!\a")
    prxxx(f"API HOST: {HOST} | PORT: {PORT}")
    config = uvicorn.Config(app, host = HOST, port=PORT, log_level="info", reload=True,ws=)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())


