import openpyxl
import os
import bs4
import urllib
import pyautogui

from selenium import *
from selenium import webdriver
from selenium .webdriver.chrome.options import Options
from keyboard import press

import time
import os
import re
import requests
import sys

#크롬 옵션 설정 함수
def give_chrome_option(folder_path):
    chromeOptions = webdriver.ChromeOptions()
    prefs = {"download.default_directory" : folder_path,
           "download.prompt_for_download": False,
           "download.directory_upgrade": True}
    chromeOptions.add_experimental_option("prefs", prefs)
    return chromeOptions

#파일 저장 경로
file_path = 'C:/Users/PC/Desktop'
chromeOptions = webdriver.ChromeOptions()
prefs = {"download.default_directory" : file_path,"download.prompt_for_download": False,"download.directory_upgrade": True}
chromeOptions.add_experimental_option("prefs", prefs)
driver = webdriver.Chrome('C:/chromedriver.exe',chrome_options=chromeOptions)

directory = "사진"

if not os.path.exists(directory):
    os.makedirs(directory)

celebrity_name = ''

# 유명인 리스트 엑셀 불러오기
read_row = 1420

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
        celebrity_name = load_ws[ 'A'+str(read_row) ].value
        print("celebrity_name : ",celebrity_name)
        
        # 사진 구글 검색 및 다운로드
        url = 'https://www.google.com/'
        driver.get(url)

        time.sleep(0.5)
        #driver.find_element_by_name('q').send_keys(celebrity_name).submit()
        elem = driver.find_element_by_name('q')
        elem.send_keys(celebrity_name)
        elem.submit()

        #press('enter')
        #driver.find_element_by_xpath("//input[@name='btnk',@value='Google 검색']").click()
        #time.sleep(0.5)
        #driver.find_element_by_xpath('//*[@id="tsf"]/div[2]/div/div[3]/center/input[1]').click()

        time.sleep(0.5)
        driver.find_element_by_link_text('이미지').click()

        time.sleep(0.5)
        images = driver.find_elements_by_tag_name('img')

        count = 0
        time.sleep(0.5)
        for image in images:
            src = image.get_attribute('src')
            #print("src : ",src)
            print(count)
            count += 1
            time.sleep(0.5)
            if count >= 3 :
                if src != None:
                    urllib.request.urlretrieve(src, "사진/"+celebrity_name+".jpg")
                    time.sleep(3)
                    break
            
    read_row += 1
