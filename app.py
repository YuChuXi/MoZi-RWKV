# -*- coding: utf-8 -*-
import time, random, re, sys, os, signal, json, tqdm
from quart import Quart, websocket, request
from hypercorn.config import Config
from hypercorn.asyncio import serve
import asyncio
from rwkv import RWKVChater, RWKVNicknameGener, RWKVGroupChater, process_default_state
from app_util import prxxx, gen_echo, clean_symbols
from typing import Dict

HOST, PORT = ("0.0.0.0", 8088)
SAVE_TIME = 3600

test_message = """告诉我关于你的一切。"""

with open("help.min.html", "r") as f:
    flask_help = f.read()

random.seed(time.time())
chaters: Dict[str, RWKVChater] = {}
group_chaters: Dict[str, RWKVGroupChater] = {}
nicknameGener = RWKVNicknameGener()

app = Quart(__name__)


def restart():
    # app.shutdown()
    python = sys.executable
    prxxx("### Restart ! ###")
    os.execl(python, python, *sys.argv)


def stop(signal=None, frame=None):
    #app.shutdown()
    prxxx("### STOP ! ###")
    sys.exit()


async def save_chaters_state():
    for id in tqdm.tqdm(chaters, desc="Save chater", leave=False, unit="chr"):
        await asyncio.sleep(0)
        await chaters[id].save_state(id, q=True)
    for id in tqdm.tqdm(
        group_chaters, desc="Save grpup chater", leave=False, unit="chr"
    ):
        await asyncio.sleep(0)
        await group_chaters[id].save_state(id, q=True)


async def time_to_save():
    while True:
        for i in range(SAVE_TIME):  # 防止卡服务器关闭
            await asyncio.sleep(1)
        await save_chaters_state()


async def chat(
    message: str,
    id: str = "-b2bi0JgEhJru87HTcRjh9vdT",
    user: str = "木子",
    nickname: str = "墨子",
    state: str = None,
) -> str:
    id = clean_symbols(id)
    echo = gen_echo()
    prxxx()
    if not id in chaters:
        chaters[id] = RWKVChater(id, state_name=state)
        await chaters[id].init_state()

    prxxx(f" #    Chat   id: {id} | user: {user} | echo: {echo}")
    prxxx(f" #    -->[{message}]-{echo}")
    answer = await chaters[id].chat(message=message, chatuser=user, nickname=nickname)
    prxxx(f" #    Chat   id: {id} | nickname: {nickname} | echo: {echo}")
    prxxx(f" #  {echo}-[{answer}]<--")

    # 如果接受到的内容为空，则给出相应的回复
    if answer.isspace() or len(answer) == 0:
        answer = "喵喵喵？"
    return answer


async def group_chat_send(
    message: str,
    id: str = "-b2bi0JgEhJru87HTcRjh9vdT",
    user: str = "木子",
    state: str = None,
) -> str:
    id = clean_symbols(id)

    if len(message) == 0:
        return

    echo = gen_echo()
    prxxx()
    if not id in group_chaters:
        group_chaters[id] = RWKVGroupChater(id, state_name=state)
        await group_chaters[id].init_state()

    prxxx(f" #    Send Gchat   id: {id} | user: {user} | echo: {echo}")
    prxxx(f" #    -->[{message}]-{echo}")
    await group_chaters[id].send_message(message=message, chatuser=user)


async def group_chat_get(
    id: str = "-b2bi0JgEhJru87HTcRjh9vdT",
    nickname: str = "墨子",
    state: str = None,
) -> str:
    id = clean_symbols(id)

    echo = gen_echo()
    prxxx()
    if not id in group_chaters:
        group_chaters[id] = RWKVGroupChater(id, state_name=state)
        await group_chaters[id].init_state()

    answer = await group_chaters[id].get_answer(nickname=nickname)
    prxxx(f" #    Get gchat   id: {id} | nickname: {nickname} | echo: {echo}")
    prxxx(f" #  {echo}-[{answer}]<--")

    # 如果接受到的内容为空，则给出相应的回复
    if answer.isspace() or len(answer) == 0:
        answer = "喵喵喵？"
    return answer


async def gen_nickname(name:str):
    echo = gen_echo()
    prxxx()
    prxxx(f" #    GenNickname   echo: {echo}")
    prxxx(f" #    -->[{name}]-{echo}")
    nickname = await nicknameGener.gen_nickname(name)
    prxxx(f" #    GenNickname   echo: {echo}")
    prxxx(f" #  {echo}-[{nickname}]<--")

    # 如果接受到的内容为空，则给出相应的回复
    if nickname.isspace() or len(nickname) == 0 or nickname == "None":
        nickname = name
    return nickname


async def reset_state(id:str):
    id=clean_symbols(id)
    flag = False
    if id in chaters:
        await chaters[id].reset_state()
        flag = True
    if id in group_chaters:
        await group_chaters[id].reset_state()
        flag = True
    return flag


@app.route("/chat", methods=["POST", "GET"])
async def R_chat():
    if request.method == "GET":
        kwargs = request.args
    elif request.method == "POST":
        kwargs = await request.form
    answer = await chat(**kwargs)
    return {"message": answer, "state": "ok"}


@app.route("/group_chat_send", methods=["POST", "GET"])
async def R_group_chat_send():
    if request.method == "GET":
        kwargs = request.args
    elif request.method == "POST":
        kwargs = await request.form
    await group_chat_send(**kwargs)
    return {"state": "ok"}


@app.route("/group_chat_get", methods=["POST", "GET"])
async def R_group_chat_get():
    if request.method == "GET":
        kwargs = request.args
    elif request.method == "POST":
        kwargs = await request.form
    answer = await group_chat_get(**kwargs)
    return {"message": answer, "state": "ok"}


@app.route("/nickname", methods=["POST", "GET"])
async def R_nickname():
    if request.method == "GET":
        kwargs = request.args
    elif request.method == "POST":
        kwargs = await request.form
    nickname = await gen_nickname(**kwargs)
    return {"nickname": nickname, "state": "ok"}


@app.route("/reset_state", methods=["GET"])
async def R_reset_state():
    if request.method == "GET":
        kwargs = request.args
    elif request.method == "POST":
        kwargs = await request.form
    flag = await reset_state(**kwargs)
    return {"state": "ok" if flag else "a?"}


@app.route("/restart", methods=["GET"])
async def R_restart():
    if request.args["passwd_gkd"] == "ihAVEcODE":
        await app.shutdown()
        restart()
    return {"state": "fuck you!"}


@app.route("/stop", methods=["GET"])
async def R_stop():
    if request.args["passwd_gkd"] == "ihAVEcODE":
        await app.shutdown()
        stop()
    return {"state": "fuck you!"}


@app.route("/", methods=["GET"])
async def R_index():
    return flask_help


@app.websocket("/chat")
async def W_chat():
    while True:
        data = json.loads(await websocket.receive())
        """
        data{
            id
            message
            username
            nickname*
            default_state*
            echo*
        }
        """
        answer = await chat(**data)
        await websocket.send(
            json.dumps({"message": answer, "state": "OK", "echo": data.get("echo", "")})
        )


@app.websocket("/group_chat")
async def W_group_chat():
    while True:
        data = json.loads(await websocket.receive())
        """
        data{
            action
            id
            message+
            username+
            nickname*
            default_state*
            echo*
        }
        """
        if data["action"] == "send":
            await group_chat_send(**data)
            await websocket.send(
                json.dumps({"state": "OK", "echo": data.get("echo", "")})
            )
        elif data["action"] == "get":
            answer = await group_chat_get(**data)
            await websocket.send(
                json.dumps(
                    {"message": answer, "state": "OK", "echo": data.get("echo", "")}
                )
            )
        else:
            await websocket.send(
                json.dumps({"state": "A?", "echo": data.get("echo", "")})
            )


# @app.before_serving
async def before_serving():
    # app.add_background_task(time_to_save)
    await process_default_state()
    await nicknameGener.init_state()
    chaters["init"] = RWKVChater("init")
    await chaters["init"].init_state()
    prxxx(f"State size: {chaters['init'].state.state.size}")
    await chaters["init"].reset_state()
    await chat(
        **{
            "id": "init",
            "message": test_message,
            "user": "测试者",
        }
    )

    prxxx()
    prxxx(" *#*   RWKV！高性能ですから!   *#*")
    prxxx()
    prxxx("Web api server start!\a")
    prxxx(f"API   HOST: {HOST} | PORT: {PORT}")


@app.after_serving
async def after_serving():
    await save_chaters_state()
    global chaters, group_chaters
    del chaters, group_chaters
    prxxx("### STOP ! ###")


async def main():
    await before_serving()  # fix: timeout wen shutup
    config = Config()
    config.bind = ["0.0.0.0:8088"]
    config.use_reloader = True
    config.loglevel = "debug"
    """
    for i in tqdm.trange(99999):
        await group_chat_send({"id":"ggtgg","message":"uuuu","user":"yyyyy"})
    """
    await serve(app, config)


if __name__ == "__main__":
    asyncio.run(main())
