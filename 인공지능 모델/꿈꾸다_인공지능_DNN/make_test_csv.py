import openpyxl
import os
import csv

directory = 'Result'

if not os.path.exists(directory):
        os.makedirs(directory)
        try:
            print("make file : "+directory)
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EXIST:
                print("file already exist : "+directory)
                raise
else :
        print("file already exist : "+directory)

for file in os.listdir("TXT위키백과_TRAIN"):
    if file.endswith(".txt"):

        # 파일 읽어서 한 변수에 저장하기
        file_name = file
        print(file_name)

        f = open("./TXT위키백과_TRAIN/"+file_name, 'r', encoding='utf-8')

        document = ""

        line = f.readline()
        document = line
        
        while line :
            line = f.readline()
            document += line
            
        f.close()

        # 엑셀 파일 읽어서 직업 번호 찾기
        file_name = file[:-4]

        wb = openpyxl.load_workbook('유명인_리스트.xlsx')
        ws = wb.get_sheet_by_name('유명인_리스트')

        job = 0

        f = open('./Result/test.csv', 'a', encoding='utf-8', newline='')

        for r in ws.rows:
            name = r[0].value
            job = r[1].value

            if( file_name == name ) :        
                # csv 파일에 쓰기
                print("write")
                print(name)
                print(job)
                print()
                
                wr = csv.writer(f)
                wr.writerow([job, document])
        
                break

        f.close()



























