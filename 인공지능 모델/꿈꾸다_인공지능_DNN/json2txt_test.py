import json
import re
import os, errno
from pprint import pprint

directory = 'TXT위키백과_TEST'

if not os.path.exists(directory):
        os.makedirs(directory)
        try:
            print("make file : "+directory)
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("file already exist : "+directory)
                raise
else :
        print("file already exist : "+directory)

# 각각의 인물들을 하나의 txt파일로 생성
for file in os.listdir("JSON위키백과_TEST"):
    if file.endswith(".json"):

        file_name = file[:-5]
        print(file_name)

        if( os.path.isfile("./TXT위키백과_TEST/"+file_name+".txt") == True ) :
            print(file_name+" : Already exist !")
            continue

        input_file = open("./JSON위키백과_TEST/" + file_name + ".json", "r", encoding="utf-8")
        output_file = open("./TXT위키백과_TEST/" + file_name + ".txt", "w", encoding="utf-8")

        while True :
            line = input_file.readline()
            
            if not line : break
            # 특수문자 제거
            line = line.replace('\\n', '')
            line = line.replace('\n', ' ')
            line = re.sub('[\W]', ' ', line)

            output_file.write(line)

input_file.close()
output_file.close()
