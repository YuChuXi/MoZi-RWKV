# -*- coding: utf-8 -*-

import time, random, re
from flask import jsonify
from waitress import serve
from flask import Flask, request
from rwkv import RWKVChater, RWKVNicknameGener, process_default_state
from app_util import prxxx, gen_echo, clean_symbols
from typing import Dict


test_msg = """告诉我关于你的一切。"""


with open("help.min.html", "r") as f:
    flask_help = f.read()

random.seed(time.time())
chaters: Dict[str,RWKVChater] = {}
process_default_state()
nicknameGener = RWKVNicknameGener()



app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    req_msg:str = request.form.get("message",default="")
    id:str = clean_symbols(request.form.get("id",default="-b2bi0JgEhJru87HTcRjh9vdT"))
    user:str = request.form.get("user",default="木子")
    nickname:str = request.form.get("nickname",default="墨子")
    multiuser:bool = request.form.get("multiuser",default=True)
    state:str = request.form.get("state",default=None)
    # req_msg = req_msg if len(req_msg) <= 256 else req_msg[:256]
    
    echo = gen_echo()
    prxxx()
    if not id in chaters:
        chaters[id] = RWKVChater(id, state_name=state)
    prxxx(f" #    Chat id:{id} user:{user} echo:{echo}")
    prxxx(f" #    -->[{req_msg}]-{echo}")
    bak_msg = chaters[id].chat(
        msg=req_msg, chatuser=user, nickname=nickname, show_user_to_model=multiuser
    )
    prxxx(f" #  {echo}-[{bak_msg}]<--")

    # 如果接受到的内容为空，则给出相应的回复
    if bak_msg.isspace() or len(bak_msg) == 0:
        bak_msg = "喵喵喵？"
    return jsonify({"message": bak_msg, "state": "ok"})


@app.route("/nickname", methods=["POST","GET"])
def nickname():
    if request.method == "POST":
        name:str = request.form.get("name",default="")
    elif request.method == "GET":
        name:str = request.args.get("name",default="")
    else:
        return "fuck you!"
    
    echo = gen_echo()
    prxxx()
    prxxx(f" #    GenNickname echo:{echo}")
    prxxx(f" #    -->[{name}]-{echo}")
    nickname = nicknameGener.gen_nickname(name)
    prxxx(f" #  {echo}-[{nickname}]<--")

    # 如果接受到的内容为空，则给出相应的回复
    if nickname.isspace() or len(nickname) == 0 or nickname == "None":
        nickname = name
    return jsonify({"nickname": nickname, "state": "ok"})



@app.route("/cleanstate", methods=["GET"])
def cleanstate():
    try:
        id:str = request.args["id"]
        if not id in chaters:
            chaters[id] = RWKVChater(id)
        chaters[id].reset()
        return jsonify({"state": "ok"})
    except:
        return """
NM
    """


@app.route("/", methods=["GET"])
def index():
    return flask_help


def test():
    chaters["init"] = RWKVChater("init")
    prxxx(f"State size:{chaters['init'].state.size}")
    prxxx(f"State shape:{chaters['init'].state.shape}")
    chaters["init"].reset()
    echo = gen_echo()
    prxxx(f" #    Test id:test user:测试者 echo:{echo}")
    prxxx(f" #    -->[{test_msg}]-{echo}")
    prxxx(f" #  {echo}-[{chaters['init'].chat(test_msg, chatuser = '测试者')}]<--")


# 启动APP
if __name__ == "__main__":
    test()
    prxxx()
    prxxx(" *#*   RWKV！高性能ですから!   *#*")
    prxxx()

    prxxx("Web api server start!\a")
    serve(app, host="0.0.0.0", port=8088,threads=8)
    # app.run(host="0.0.0.0", port=8088, debug=False)
