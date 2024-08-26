from tkinter import *
from tkVideoPlayer import TkinterVideo
from playsound import playsound
# filedialog函数用于上传文件
from tkinter import filedialog
# 因为头像是图片需要用到pillow库
from PIL import Image

# 创建窗口
win = Tk()
# 设置窗口标题
win.title('虚拟人A')
# 设置窗口宽度和高度
win.geometry('1920x1080')
#淡蓝色背景
win.config(bg="#b0e0e6")

#视频窗口
video=LabelFrame(win,text="video",relief="solid", bd=2,height=700,width=1000,bg="black")
video.config(bg="black")
video.pack()

answer=Canvas(video,height=700,width=1000,bg="red")
answer.pack()

videoplayer = TkinterVideo(master=answer, scaled=True,height=400,width=400)
videoplayer.load(r"../result/video/background_image1##answer_enhanced.mp4")
videoplayer.pack(expand=True, fill="both")
videoplayer.play()  # play the video


#聊天窗口
talk=LabelFrame(win,text="talk",relief="solid", bd=2,height=50,width=400)
talk.config(bg="yellow")
talk.pack()

#输入框以及按钮
input=Entry(talk,bd=2,width=80)
input.pack()

def printget():
    print(input.get())
Button(talk, text="输入", command=printget).pack()



# 主循环
win.mainloop()