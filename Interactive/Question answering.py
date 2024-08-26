   # 打开第一个 txt 文件，并读取其中的内容
with open("../result/voice_input.txt", "r",encoding="utf-8") as f:
    content = f.read()

# 判断内容是否包含"你好啊"
if content.find("你好啊") > -1:
    # 输出内容到第二个 txt 文件中
    with open("../result/answer.txt", "w",encoding="utf-8") as f:
        f.write("我是卖货机器人，欢迎选购我们的商品呀\n")