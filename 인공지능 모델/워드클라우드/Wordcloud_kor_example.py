from collections import Counter
from konlpy.tag import Twitter
import pytagcloud
 
f = open('블라디미르_푸틴.txt','r',encoding='utf-8')
data = f.read()
 
nlp = Twitter()
nouns = nlp.nouns(data)
 
count = Counter(nouns)
tags2 = count.most_common(40)
taglist = pytagcloud.make_tags(tags2, maxsize=80)
pytagcloud.create_tag_image(taglist, 'Wordcloud_kor_example.jpg', size=(900, 600), fontname='Nanum Gothic', rectangular=False)
 
f.close()
