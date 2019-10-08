1. 위키백과 크롤링.py, 위키백과 크롤링(테스트).py 를 통해 데이터를 수집한다.

=> training data와 test data를 나눈다.

2. json2txt.py 를 통해서 json 파일을 txt파일로 변경한다.

3. make_test_csv.py 를 통해서 train data를 모두 csv파일로 만들어 준다.

(교사학습을 위한 작업)

4. _makemodel.py 를 통해서 test.csv 파일로 model과 dictionary를 만든다.

5. _loadmodel.py 를 통해서 test 폴더 안에 있는 모든 txt파일을 테스트 해본다.

