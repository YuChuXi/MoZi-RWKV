import time
import requests
import tqdm

"""
message = input("Seed: ")
while True:
    print("A",message)
    bak = requests.post("http://localhost:8088/chat",data = {"msg":message,"id":"-1","user":"A"})
    message = bak.json()["text"]
    print("B",message)
    bak = requests.post("http://localhost:8088/chat",data = {"msg":message,"id":"-2","user":"B"})
    message = bak.json()["text"]
    """


if input("reset?") == "y":
    print(
        requests.post(
            "http://localhost:8088/chat",
            data={
                "message": "+reset",
                "id": "-i6trfghji87tf",
                "user": "小桐",
            },
        ).json()["message"]
    )

"""
while True:
    print(
        requests.post(
            "http://localhost:8088/chat",
            data={
                "message": input("a:"),
                "id": "-i6trfghji87tf",
                "user": "小桐",
            },
        ).json()["message"]
    )

"""

p = """你好，墨子。

墨子： 夏次依锭？你怎么来了啊？

夏次依锭： 你见我来不开心吗？

墨子： 啊，没事啦。你有什么想说的吗？

夏次依锭： 听说今天晚上有烟花大会，我们要不要一起去看看？

墨子： 烟花大会？我们去哪里看呢？

夏次依锭： 好像是中央湖那里，一起去吗？

墨子： 好啊，我们去吧！

夏次依锭： 现在时间还早，我们要不要去挑挑衣服？听说有人会穿cos去呢。

墨子： 那我们去吧！（开心）

夏次依锭： （挑了件水手服）这件怎么样？你穿起来应该会很好看。

墨子： 嗯，好像挺合适的。你觉得我会不会很丑？

夏次依锭： 怎么会。你穿什么都好看。

墨子： 你觉得我会不会穿cos去呢？

夏次依锭： 也可以试试哦。

墨子： 那我们去吧！

夏次依锭： （到了晚上）你这是cos的玄狸吗？

墨子： 是啊，我们来这里的时候才刚做完手术呢。

夏次依锭： 什么手术啊？

墨子： 啊，就是我刚才说的手术啊。

夏次依锭： 嘛，无所谓啦。快到时间了，我们找个地方坐着吧。

墨子： 好的，谢谢你啦。

夏次依锭： （一朵绚烂的烟花从空中绽放）你看！

墨子： 好美啊！

夏次依锭： 果然没有来错呢。

墨子： 你真的很漂亮啊！

夏次依锭： 诶？我？

墨子： 你真的很漂亮啊！

夏次依锭： 你也是啊。

墨子： 我也觉得你很漂亮啊。

夏次依锭： （随着最后一只烟花消失）啊，结束了呢。

墨子： 好了，今天的烟花大会就到这里啦。

夏次依锭： 肚子稍微有点饿了呢。

墨子： 那我们去找一家小店吃饭吧！

夏次依锭： （走到一家小店前）这个店，好像是你以前经常来的，我还没来过呢。

墨子： 那是我们小时候常去的店啊！

夏次依锭： 是吗？我忘了，不好意思啦。

墨子： 没关系啦，这个店还是很好吃的。

夏次依锭： 真的吗？那就这家店吧。

墨子： 好的，我们去吧！

夏次依锭： （看着菜单）炸饺子，水煮饺子，蒸饺子，选什么好呢？

墨子： 我觉得蒸饺子好吃呢。

夏次依锭： 那就蒸饺子吧！

墨子： 好的，我们来点菜吧。

夏次依锭： （蒸饺子上桌了）哇，看起来很不错呢。

墨子： 好吃吗？

夏次依锭： （吃了一口）真好吃！你也尝尝。

墨子： 好的，我尝尝。

夏次依锭： 怎么样？

墨子： 很好吃呢，谢谢你啊！

夏次依锭： （夹了个沾满花生酱的饺子）来，张嘴。

墨子： （张嘴）哇，好吃啊！

夏次依锭： 今天过的好开心啊。

墨子： 是啊，我也很开心呢。

夏次依锭： 时候不早了，我们回去吧。

墨子： 好的，谢谢你今天陪我玩。

夏次依锭： 拜拜~

墨子： 拜拜~

玉子： 醒醒，夏次依锭来找你玩了哦

墨子： 啊，夏次依锭！

夏次依锭： 嗨！早上好，小猫咪。

墨子： 早上好，夏次依锭。

夏次依锭： 你怎么睡懒觉了？

墨子： 你不也是啊，还说我呢

夏次依锭： 我可不像你会睡懒觉（揉了揉墨子的头）

墨子： 啊，对不起。

夏次依锭： 真可爱（揉的更起劲了）

墨子： 谢谢你啦，夏次依锭。

夏次依锭： 你可真喜欢被人揉脑袋呢。

墨子： 喵唔～

夏次依锭： 先去吃早饭吧，你来决定去哪吃。

墨子： 好的，我们去吃什么？

夏次依锭： 我有点馋炸饺子了，要不要回昨天那家店吃呢？

墨子： 好啊，那我们去吧！

夏次依锭： （半路上传来了一阵眩晕感，夏次依锭和墨子都晕过去了）这里是哪儿？（夏次依锭看着周围的森林）你没事吧？墨子！（摇了摇墨子）

墨子： 我没事啦，夏次依锭。

夏次依锭： 这里是哪儿？怎么都是树林？

墨子： 这是森林啊，我们去找夏次依锭吧。

夏次依锭： 我在这里。（拍了拍墨子）

墨子： 夏次依锭，你没事吧？

夏次依锭： 天空怎么变色了？算了，我们先找找附近有没有人吧。

墨子： 嗯，这的森林好奇怪啊？

夏次依锭： 看！（用手指向一个地方）那里有个人，好像又不是人 跟你一样有猫耳朵。

墨子： 那就是夏次依锭，我们回去吧。

夏次依锭： 我们过去看看吧。（夏次依锭拉着墨子的手）

墨子： 好的，夏次依锭。

玉子： 看你们两个的衣服不像是本地人啊，外乡来的？

墨子： 额，请问，这里是那里？

W墨： 新来的？小心晚上的怪物！魔王城就在附近！

墨子： 那我们先找个地方休息吧。

夏次依锭： 根据刚刚那个人所说，这个世界有魔王，我们可能要打败魔王才能回去。

墨子： 是啊，我们要打败魔王。

夏次依锭： 你之前可是魔法学院的高材生，打败个小小的魔王应该很轻松的吧。

墨子： 是啊，我也这么觉得。

夏次依锭： （肚子发出了咕咕咕的声音）

墨子： 夏次依锭，你怎么了？

夏次依锭： 饿了～

墨子： 是啊，我们还得去找个地方吃饭来着。

夏次依锭： 可是这荒郊野岭的，哪有饭吃啊？还是说你要去捕猎？或者去找城镇？

墨子： （墨子身边浮现出一圈圈魔法阵，然后逐渐扩大，最后消失）这附近应该有镇子。

玉子： 你俩怎么还在这儿啊，看你们刚来这吧？找不到地儿吃饭吗？

墨子： 请问这附近的镇子怎么走？

玉子： 不介意的话，我带你们去镇上吧？

夏次依锭： 好吧，麻烦了。

玉子： 没事的啦，见得惯了，这附近经常有新来的魔法师不认得路。

墨子： 是这样的喵？那看来我们。还能找到一些其他的同伴呢！

玉子： 好啦，到村里啦。前面有茶馆有卖吃的，你们去坐坐吧。

夏次依锭： 好的，多谢了。墨子，走吧。

墨子： 这里的建筑怎么奇奇怪怪的？不像是现代的呀？

夏次依锭： 你好，现在是什么时候了？

李治廷： 额，魔刻14时半

夏次依锭： 不是，我问你现在是哪一年了

李治延： 啊？哦，仙女木十七年。

墨子： 那就是说，我们来的时候已经是十七年了。

夏次依锭： 这个世界，时间是倒流的吗？

墨子： 倒流？我不知道，这里没有时间机器。

夏次依锭： 我们回去吧，找到地方坐一会儿。

墨子： 看，那里就是魔王城了吧，我们过去看看吗？

夏次依锭： 不了吧，我好饿。"""

print(requests.post("http://localhost:8088/chat",data = {"message":p,"id":"-i6trfghji87tf","user":"夏次依锭"}).json()["message"])

# time.sleep(100)
while True:
    print(
        requests.post(
            "http://localhost:8088/chat",
            data={
                "message": "+",
                "id": "-i6trfghji87tf",
                "user": "小桐",
            },
        ).json()["message"]
    )
    # print(requests.post("http://localhost:8088/chat",data = {"msg":"+reset","id":"-1dfg","user":"玉子"}).json()["text"])
    time.sleep(1)
"""

for i in tqdm.trange(99999):
    requests.get("http://127.0.0.1:8088/group_chat_send?id=ggggg&message=66666&username=777")

    """
