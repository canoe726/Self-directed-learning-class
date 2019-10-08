import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from konlpy.tag import Twitter; t = Twitter()

import nltk

directory = 'WordCloud'

if not os.path.exists(directory):
        os.makedirs(directory)
        try:
            print("make file : "+directory)
            os.makedirs(directory)
        except OSError:
            print("file already exist : "+directory)
else :
        print("file already exist : "+directory)

stop_words = []

with open('stopwords.txt', 'r') as f:
    for line in f:
        stop_words.append(line[:-1])

# print(stop_words)

for file in os.listdir("위키백과TXT"):
    if file.endswith(".txt"):

        file_name = file[:-4]
        # print(file_name)

        # 이미 워드 클라우드가 있으면 생성 안함
        if( os.path.isfile("./WordCloud/"+file_name+".jpg") == True ) :
            print(file_name+" : Already exist !")
            continue

        # 워드클라우드가 없으면 생성
        if( os.path.isfile("./WordCloud/"+file_name+".jpg") == True ) :
            print(file_name+" : Already exist !")
            continue
          
        print(file_name + ".jpg")

        input_file = open("./위키백과TXT/" + file_name + ".txt", "r", encoding="utf-8")

        ko_con_text = input_file.read()

        token_ko = t.nouns(ko_con_text)

        token_ko = [each_word for each_word in token_ko if each_word not in stop_words]

        ko = nltk.Text(token_ko, name=file_name)
        print(ko.vocab().most_common(5))

        data = ko.vocab().most_common(20)
        tmp_data = dict(data)

        font_path = 'c:/Windows/Fonts/malgun.ttf'

        constitution_coloring = np.array(Image.open("./사진/blue_cloud.jpg"))
        image_colors = ImageColorGenerator(constitution_coloring)

        wordcloud = WordCloud(font_path=font_path, relative_scaling=0.2,
                              mask=constitution_coloring, background_color='white',
                              min_font_size=1, max_font_size=40).generate_from_frequencies(tmp_data)

        # recolor는 이미지 색에 맞춰서 글자색 변경
        #plt.figure(figsize=(12,12))
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
        #plt.axis("off")
        #plt.show()

        wordcloud.to_file("./WordCloud/" + file_name + '.jpg')
        print("saved!")

