from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from konlpy.tag import Twitter; t = Twitter()

import nltk


f = open('대한민국헌법.txt')

ko_con_text = f.read()

token_ko = t.nouns(ko_con_text)

stop_words = ['제','월','일','조','수','때','그','이','바','및','안']

token_ko = [each_word for each_word in token_ko if each_word not in stop_words]

ko = nltk.Text(token_ko, name='대한민국헌법')
print(ko.vocab().most_common(50))

data = ko.vocab().most_common(500)
tmp_data = dict(data)

font_path = 'c:/Windows/Fonts/malgun.ttf'

constitution_coloring = np.array(Image.open("헌법.jpg"))
image_colors = ImageColorGenerator(constitution_coloring)

wordcloud = WordCloud(font_path=font_path, relative_scaling=0.2,
                      mask=constitution_coloring, background_color='white',
                      min_font_size=1, max_font_size=40).generate_from_frequencies(tmp_data)

# recolor는 이미지 색에 맞춰서 글자색 변경
plt.figure(figsize=(12,12))
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file('Wordcloud_kor_image.jpg')


