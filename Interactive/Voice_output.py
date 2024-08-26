#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
from aip import AipSpeech
from pydub import AudioSegment

# Step 1, Using baidu AI to generate mp3 file from text
# input your APP_ID/API_KEY/SECRET_KEY
APP_ID = '33857383'
API_KEY = 'PRYLs6vn2ZedNlYSeZp8nhT0'
SECRET_KEY = 'WbnYIeFv7lM4TO9NC32BEK1GhBjAzokI'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

with open("../result/answer.txt", "r",encoding="utf-8") as file:
    content = file.read()

result = client.synthesis(content, 'zh', 1, {'vol': 5, 'per': 4})

if not isinstance(result, dict):
    with open('../input/mp3/test.mp3', 'wb') as f:
        f.write(result)

# Step 2, convert the mp3 file to wav file
sound = AudioSegment.from_mp3('test.mp3')
sound.export("../input/audio/answer.wav", format="wav")