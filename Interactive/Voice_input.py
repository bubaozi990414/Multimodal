import codecs

text = input("请输入文本:")
with codecs.open("../result/voice_input.txt", "w", "utf-8-sig") as file:
    file.write(text)

file.close()