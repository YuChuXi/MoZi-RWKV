import httpx
import random
import time

random.seed(time.time())

username = "小小猫"
id = "-45f59d44"

if input("reset(Y/n)? ").lower() == "y":
    print(
        httpx.get(f"http://127.0.0.1:8088/reset_state?id={id}", timeout=99999).text
    )

while True:
    seq = input(f"{username}\t>> ")
    print(f"墨子\t>> ",end="",flush=True)
    req = httpx.post(
        "http://127.0.0.1:8088/chat",
        data={
            "message": seq,
            "id": id,
            "user": username,
            "state": "default.state",
            "debug": True 
        },
        timeout=99999,
    ).json()["message"]
    for i in req:
        print(i,end="",flush=True)
        time.sleep(0.066)
    print("")
