import json
import numpy as np
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
import os
import openpyxl

# we're still going to use a Tokenizer here, but we don't need to fit it
tokenizer = Tokenizer(num_words=5000)

# for human-friendly printing
case_number = 19

labels = ['official', 'musician', 'comedian', 'actor', 'soccer',
          'basketball',  'Baseball', 'athlete', 'poet', 'soldier',
          'businessmen', 'scholar', 'player', 'moviedirector', 'progamer',
          'religionist', 'journalist', 'artist', 'author']

labels_ko = ['공무원', '음악인', '개그맨', '배우', '축구',
          '농구',  '야구', '운동선수', '시인', '군인',
          '기업인', '학자', '연주가', '영화감독', '프로게이머',
          '종교인', '언론인', '예술가', '작가']

# read in our saved dictionary
with open('./Result/dictionary.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)

# this utility makes sure that all the words in your input
# are registered in the dictionary
# before trying to turn them into a matrix.
def convert_text_to_index_array(text):
  
    words = kpt.text_to_word_sequence(text)
    wordIndices = []

    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word])
        #else:
            #print("'%s' not in training corpus; ignoring." %(word))
            
    return wordIndices

# read in your saved model structure
json_file = open('./Result/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# and create a model from that
model = model_from_json(loaded_model_json)

# and weight your nodes with your saved values
model.load_weights('./Result/model.h5')

# define accuracy matrix
accuracy_matrix = [[0]*case_number for i in range(case_number)]

# count total test case file
total_test_case = 0

for file in os.listdir("./TXT위키백과_TEST") :
  if file.endswith(".txt") :
    total_test_case += 1

print("total_test_case : ", total_test_case, "\n")

# load test case .txt file
for file in os.listdir("./TXT위키백과_TEST") :
  if file.endswith(".txt") :
    
    file_name = file
    f = open("./TXT위키백과_TEST/"+file_name, 'r', encoding='utf-8')
  
    file_name = file_name[:-4]
    #print("Person : " + file_name)
  
    evalSentence = ""
  
    # read all lines
    line = f.readline()
    evalSentence = line
  
    while line :
      line = f.readline()
      evalSentence += line
    
    f.close()

    # format your input for the neural net
    testArr = convert_text_to_index_array(evalSentence)
    input = tokenizer.sequences_to_matrix([testArr], mode='binary')
    
    # predict which bucket your input belongs in
    pred = model.predict(input)
    
    # and print it for the humans
    #print("Result = Job : %s ; Confidence : %f% %\n" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))
    
    # add accuracy matrix
    wb = openpyxl.load_workbook('유명인_리스트.xlsx')
    ws = wb.get_sheet_by_name('테스트_리스트')
    
    for r in ws.rows:
      name = r[0].value
      
      if( name == file_name ):     
        job = r[1].value        
        labels_index = 0
        
        for i in range(case_number) :
          if( labels[i] == labels[np.argmax(pred)] ) :
            labels_index = i
            break
        
        # row : correct , col : prediction
        accuracy_matrix[job][labels_index] += 1
    
# print accuracy matrix
correct = 0

print("-----Accuracy Matrix-----")
for i in range(-1, case_number) :
  for j in range(-1, case_number) :
    
    if( i == j ) : correct += accuracy_matrix[i][j]
    
    if i == -1 :
      if j == -1 :
          print("C\P ",end='')
          continue
      if j == case_number :
          print()
          break
      print(labels[j][:3]+" ", end='')
    
    else :
      if j == -1 :
          print(labels[i][:3], end='')
          continue
      if j == case_number :
          print()
          break
      print("%4d"%accuracy_matrix[i][j], end='')
        
  print()  
    
print('\nTotal Accuracy : ', correct / total_test_case)








    
