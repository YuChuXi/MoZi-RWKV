import httpx
import random
import time

random.seed(time.time())

username = "小小猫"
id = "-45f59d44"

if input("reset(Y/n)? ").lower() == "y":
    print(
        httpx.get(f"http://1.13.16.50:18082/cleanstate?id={id}", timeout=1e9).text
    )

while True:
    seq = input(f"{username}\t>> ")
    print(f"墨子\t>> ?",end="",flush=True)
    req = httpx.post(
        "http://1.13.16.50:18082/message",
        data={
            "msg": seq,
            "id": id,
            "user": username,
            "state": "default.state",
        },
        timeout=1e9,
    ).json()["text"]
    for i in req:
        print(i,end="",flush=True)
        time.sleep(0.066)
    print("")
