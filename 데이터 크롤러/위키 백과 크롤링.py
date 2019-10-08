import openpyxl
import os

import json
from collections import OrderedDict

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
import urllib.request

# 내용, 테이블 찾기 함수
def contents_search(header):
    result = ""
    
    if (header.name == "h2" or header.name == "h3"
        or header.name == "h4" or header.name == "h5" or header.name == "h6"):
        nextNode = header
        # print(header.name)

        while True:
            nextNode = nextNode.nextSibling
            
            if nextNode is None:
                break
            
            # 본문 내용
            if isinstance(nextNode, NavigableString):
                    nextNode.strip()
            if isinstance(nextNode, Tag):        
                if (nextNode.name == "h2" or nextNode.name == "h3"
                    or nextNode.name == "h4" or nextNode.name == "h5" or nextNode.name == "h6"):
                    break                
                #elif nextNode.name == 'table':
                #    continue
                else :
                    for e in nextNode :
                        if e.name == 'sup' :
                            continue
                        elif e.name == 'a' :
                            result += e.text
                        elif e.name == None :
                            result += e
                        else :
                            result += e.text + ' '
                            
                    result += '<br><br>'     

    return result


# 유명인 위키 백과 검색 함수
def spider(name):

    infobox_vcard_exist = False
    infobox_biography_vcard_exist = False
    
    url = "https://ko.wikipedia.org/wiki/"

    req = requests.get(url+name)
    text = req.text
    text_soup = BeautifulSoup(text, 'lxml')

    # 인물 기본 내용 추출
    name_content = ''

    name_tag = text_soup.find('div', {'class':'mw-parser-output'})
    for name_con in name_tag:
        if name_con.name == 'h2' :
            break
        if name_con.name == 'p' :
            name_content += name_con.text

    if name_content != '' :
        celebrity_data[ name+' 내용' ] = name_content

    # infobox biography vcard 일 경우
    if text_soup.find('table', {'class':'infobox biography vcard'}) :
        infobox = text_soup.find('table', {'class':'infobox biography vcard'})

        for item in infobox.findAll('tr'):
                
            infobox_key = item.find('th') # , attrs={'style':'text-align:left;'}
                
            if infobox_key != None :
                infobox_value = item.find('td',attrs={'class':'','style':''})
                if infobox_value != None :
                    celebrity_data[ infobox_key.get_text() ] = infobox_value.get_text()
                    infobox_biography_vcard_exist = True
                    infobox_vcard_exist = True

    # infobox vcard 일 경우
    if infobox_biography_vcard_exist == False :
        if text_soup.find('table', {'class':'infobox vcard'}) :
            infobox = text_soup.find('table', {'class':'infobox vcard'})

            for item in infobox.findAll('tr'):
                
                infobox_key = item.find('th') # , attrs={'style':'text-align:left;'}
                
                if infobox_key != None :
                    infobox_value = item.find('td',attrs={'class':'','style':''})
                    if infobox_value != None :
                        celebrity_data[ infobox_key.get_text() ] = infobox_value.get_text()
                        infobox_biography_vcard_exist = True
                        infobox_vcard_exist = True

    # infobox 일 경우
    if infobox_vcard_exist == False and infobox_biography_vcard_exist == False :
        infobox = text_soup.find('table', {'class':'infobox'})

        if( infobox != None ) :
            for item in infobox.findAll('tr'):
                
                infobox_key = item.find('th') # , attrs={'style':'text-align: left'}
                #print(infobox_key)
                
                if infobox_key != None :
                    infobox_value = item.find('td')
                    if infobox_value != None :
                        celebrity_data[ infobox_key.get_text() ] = infobox_value.get_text()
        

    h2_key = ""
    h3_key = ""
    h4_key = ""
    h5_key = ""
    h6_key = ""
    
    # 내용 검색
    for contents in text_soup.findAll({"h2","h3","h4","h5","h6"}):

        if contents.text == "목차":
            continue
        if contents.text == "범례[편집]" or contents.text == "범례" :
            continue
        if contents.text == "각주[편집]" or contents.text == "각주" :
            break
        if contents.text == "같이 보기[편집]" or contents.text == "같이 보기" :
            break
        if contents.text == "외부 링크[편집]" or contents.text == "외부 링크" :
            break

        key = contents.text
        if any( "[편집]" in key for k in key ) :
            key = key[0:-4]
        #print(key)
        
        h2_content = ""
        h3_content = ""
        h4_content = ""
        h5_content = ""
        h6_content = ""

        # 테이블 1
        data = []
        wikitable = text_soup.find("table",{"class":"wikitable"})
        
        # 테이블 2
        data2 = []
        wikitable_sortable = text_soup.find("table",{"class":"wikitable sortable"})

        # 본문 내용
        if contents.name == "h2":
            h2_key = key
            h3_key = ""
            h4_key = ""
            h5_key = ""
            h6_key = ""

            h2_content = contents_search(contents)
            celebrity_data[ h2_key ] = {}            
            
        if contents.name == "h3":
            h3_key = key
            h4_key = ""
            h5_key = ""
            h6_key = ""

            # 위키백과 오류, h2 없음
            if( h2_key == "" ) :
                h2_key = "내용"
                celebrity_data[ h2_key ] = {}

            h3_content = contents_search(contents)
            celebrity_data[ h2_key ][ h3_key ] = {}
            
        if contents.name == "h4":
            h4_key = key
            h5_key = ""
            h6_key = ""

            # 위키백과 오류, h2 -> h4
            if( h3_key == "" ) :
                h3_key = "내용"
                celebrity_data[ h2_key ][ h3_key ] = {}

            h4_content = contents_search(contents)
            celebrity_data[ h2_key ][ h3_key ][ h4_key ] = {}
            

        if contents.name == "h5":
            h5_key = key
            h6_key = ""
            
            # 위키백과 오류, h2 -> h5
            if( h3_key == "" ) :
                h3_key = "내용"
                celebrity_data[ h2_key ][ h3_key ] = {}

            # 위키백과 오류, h3 -> h5
            if( h4_key == "" ) :
                h4_key = "내용"
                celebrity_data[ h2_key ][ h3_key ][ h4_key ] = {}
            
            h5_content = contents_search(contents)
            celebrity_data[ h2_key ][ h3_key ][ h4_key ][ h5_key ] = {}

        if contents.name == "h6":
            h6_key = key

            # 위키백과 오류, h2 -> h6
            if( h3_key == "" ) :
                h3_key = "내용"
                h4_key = "내용"
                h5_key = "내용"
                
                celebrity_data[ h2_key ][ h3_key ] = {}
                celebrity_data[ h2_key ][ h3_key ][ h4_key ] = {}
                celebrity_data[ h2_key ][ h3_key ][ h4_key ][ h5_key ] = {}

            # 위키백과 오류, h3 -> h6
            if( h4_key == "" ) :
                h4_key = "내용"
                h5_key = "내용"
                
                celebrity_data[ h2_key ][ h3_key ][ h4_key ] = {}
                celebrity_data[ h2_key ][ h3_key ][ h4_key ][ h5_key ] = {}

            # 위키백과 오류, h4 -> h6
            if( h5_key == "" ) :
                h5_key = "내용"
             
                celebrity_data[ h2_key ][ h3_key ][ h4_key ][ h5_key ] = {}

            h6_content = contents_search(contents)
            celebrity_data[ h2_key ][ h3_key ][ h4_key ][ h5_key ][ h6_key ] = {}

        # 본문 내용 저장
        if h2_key != "" and h2_content != "":
            celebrity_data[ h2_key ].update( { '내용' : h2_content } )
            continue

        if h3_key != "" and h3_content != "":
            celebrity_data[ h2_key ][ h3_key ].update( { '내용' : h3_content } )
            continue

        if h4_key != "" and h4_content != "":
            celebrity_data[ h2_key ][ h3_key ][ h4_key ].update( { '내용' : h4_content } )
            continue

        if h5_key != "" and h5_content != "":
            celebrity_data[ h2_key ][ h3_key ][ h4_key ][ h5_key ].update( { '내용' : h5_content } )

        if h6_key != "" and h6_content != "":
            celebrity_data[ h2_key ][ h3_key ][ h4_key ][ h5_key ][ h6_key ].update( { '내용' : h6_content } )

# 유명인 리스트 엑셀 불러오기
read_row = 3252

directory = "위키백과"

if not os.path.exists(directory):
    os.makedirs(directory)

while True :
    # 파일 불러오기
    load_wb = openpyxl.load_workbook("유명인_리스트.xlsx")
    load_ws = load_wb.get_sheet_by_name("유명인_리스트")

    if load_ws[ 'A'+str(read_row) ].value == None :
        print("Last Row Number : ",read_row)
        print("Finish!!")
        break
    else :
        print("Current Row Number : ",read_row)
        
        # 함수 호출
        celebrity_data = {}

        celebrity_name = load_ws[ 'A'+str(read_row) ].value

        if( os.path.isfile("위키백과/"+celebrity_name+".json") == True ) :
            print(celebrity_name+" : Already exist !")
            read_row += 1
            continue

        celebrity_data.update({'이름':celebrity_name})
        
        print(celebrity_name)
        spider(celebrity_name)

        # Print JSON
        # print(json.dumps(celebrity_data, ensure_ascii=False, indent="\t") )

        write_file_name = celebrity_name + '.json'

        celebrity_data = [celebrity_data]

        # Write JSON
        with open("위키백과/"+write_file_name, 'w', encoding="utf-8") as make_file:
            json.dump(celebrity_data, make_file, ensure_ascii=False, indent="\t")
            
    read_row += 1
