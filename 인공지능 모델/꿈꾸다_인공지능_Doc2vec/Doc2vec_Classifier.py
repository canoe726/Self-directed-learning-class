import logging
import random

import numpy as np
import pandas as pd
import keras
from gensim.models import doc2vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

labels = ['official', 'musician', 'comedian', 'actor', 'soccer',
          'basketball',  'Baseball', 'athlete', 'poet', 'soldier',
          'businessmen', 'scholar', 'player', 'moviedirector', 'progamer',
          'religionist', 'journalist', 'artist', 'author']

labels_ko = ['공무원', '음악인', '개그맨', '배우', '축구',
          '농구',  '야구', '운동선수', '시인', '군인',
          '기업인', '학자', '연주가', '영화감독', '프로게이머',
          '종교인', '언론인', '예술가', '작가']

case_number = 19
accuracy_matrix = [[0]*case_number for i in range(case_number)]
    
num_of_epoch = 100
size_of_vector = 500
size_of_test = 0.20

def read_dataset(path):
    dataset = pd.read_csv(path, header=0, sep=",", names=['job', 'contents'])
    x_train, x_test, y_train, y_test = train_test_split(dataset.contents, dataset.job,
                                                        random_state=0, test_size=size_of_test)
    x_train = label_sentences(x_train, 'Train')
    x_test = label_sentences(x_test, 'Test')
    all_data = x_train + x_test
    return x_train, x_test, y_train, y_test, all_data


def label_sentences(corpus, label_type):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the LabeledSentence method. The format will be "TRAIN_i" or "TEST_i" where "i" is
    a dummy index of the review.
    """
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(doc2vec.LabeledSentence(v.split(), [label]))
    return labeled


def get_vectors(doc2vec_model, corpus_size, vectors_size, vectors_type):
    """
    Get vectors from trained doc2vec model
    :param doc2vec_model: Trained Doc2Vec model
    :param corpus_size: Size of the data
    :param vectors_size: Size of the embedding vectors
    :param vectors_type: Training or Testing vectors
    :return: list of vectors
    """
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = doc2vec_model.docvecs[prefix]
    return vectors


def train_doc2vec(corpus):
    logging.info("Building Doc2Vec vocabulary")
    d2v = doc2vec.Doc2Vec(min_count=5,  # Ignores all words with total frequency lower than this
                          window=10,  # The maximum distance between the current and predicted word within a sentence
                          vector_size=size_of_vector,  # Dimensionality of the generated feature vectors
                          workers=5,  # Number of worker threads to train the model
                          alpha=0.025,  # The initial learning rate
                          min_alpha=0.00025,  # Learning rate will linearly drop to min_alpha as training progresses
                          dm=1)  # dm defines the training algorithm. If dm=1 means ‘distributed memory’ (PV-DM)
                                 # and dm =0 means ‘distributed bag of words’ (PV-DBOW)
    d2v.build_vocab(corpus)

    logging.info("Training Doc2Vec model")
    # 10 epochs take around 10 minutes on my machine (i7), if you have more time/computational power make it 20
    for epoch in range(num_of_epoch):
        logging.info('Training iteration #{0}'.format(epoch))
        d2v.train(corpus, total_examples=d2v.corpus_count, epochs=d2v.iter)
        # shuffle the corpus
        random.shuffle(corpus)
        # decrease the learning rate
        d2v.alpha -= 0.0002
        # fix the learning rate, no decay
        d2v.min_alpha = d2v.alpha

    logging.info("Saving trained Doc2Vec model")
    d2v.save("d2v.model")
    return d2v


def train_classifier(d2v, training_vectors, training_labels):
  
    logging.info("Classifier training")
    train_vectors = get_vectors(d2v, len(training_vectors), size_of_vector, 'Train')
    training_labels = keras.utils.to_categorical(training_labels, case_number)
    print(len(training_labels),len(training_labels[0]))
   
    model = Sequential()
        
    model.add(Dense(1024, input_shape=(size_of_vector,), activation='sigmoid'))
    model.add(Dropout(0.5))
    
    model.add(Dense(512, input_shape=(size_of_vector,), activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, input_shape=(size_of_vector,), activation='sigmoid'))
    model.add(Dropout(0.5))

    model.add(Dense(case_number, activation='softmax'))


    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    model.fit(train_vectors, training_labels, batch_size=512, epochs=num_of_epoch,
          
          verbose=1, validation_split=0.1, shuffle=True)

    model_json = model.to_json()

    with open('./model.json', 'w') as json_file:
        json_file.write(model_json)

    model.save_weights('./model.h5')

    '''
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

    #model = LogisticRegression()
    model.fit(train_vectors, np.array(training_labels))
    training_predictions = model.predict(train_vectors)
    logging.info('Training predicted classes: {}'.format(np.unique(training_predictions)))
    logging.info('Training accuracy: {}'.format(accuracy_score(training_labels, training_predictions)))
    logging.info('Training F1 score: {}'.format(f1_score(training_labels, training_predictions, average='weighted')))
    '''
    return model


def test_classifier(d2v, classifier, testing_vectors, testing_labels):
    logging.info("Classifier testing")
    test_vectors = get_vectors(d2v, len(testing_vectors), size_of_vector, 'Test')
    testing_predictions = classifier.predict(test_vectors)
    logging.info('Testing predicted classes: {}'.format(np.unique(testing_predictions)))
    
    #logging.info('Testing accuracy: {}'.format(accuracy_score(testing_labels, testing_predictions)))
    #logging.info('Testing F1 score: {}'.format(f1_score(testing_labels, testing_predictions, average='weighted')))
    
    # accuracy matrix start    
    test_dataset = []
    for i in testing_labels :
      test_dataset.append(i)
    
    test_dataset_len = len(test_dataset)
    
    print("Length of whole` dataset : ", test_dataset_len / size_of_test)
    print("Length of training dataset : ", (test_dataset_len / size_of_test) - test_dataset_len)
    print("Length of test dataset : ",test_dataset_len,"\n")
    
    #print("test_dataset : ",test_dataset)
    
    prediction = []
    for rate in testing_predictions:
      index = 0
      max_rate = 0
      max_index = 0
      
      for i in rate:
        if( max_rate < i ) :
          max_rate = i
          max_index = index
        index += 1
      
      prediction.append(max_index)
    
    #print("prediction : ",prediction)

    for i in range(test_dataset_len) :
      if( test_dataset[i] == prediction[i] ) :
        accuracy_matrix[ test_dataset[i] - 1 ][ prediction[i] - 1 ] += 1
      else :
        accuracy_matrix[ test_dataset[i] - 1 ][ prediction[i] - 1 ] += 1
    
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
    
    """
    print("-----Accuracy Matrix-----\n")
    print("row : correct data, column : predict data\n")
    for i in range(case_number) :
      for j in range(case_number) :
        if( i == j ) :
          correct += accuracy_matrix[i][j]
        print(accuracy_matrix[i][j]," ", end='')
      print()  
    """
    print('\nTotal Accuracy : ', correct / test_dataset_len)
     # accuracy matrix end

if __name__ == "__main__":
    path = './Result/test.csv'  
    x_train, x_test, y_train, y_test, all_data = read_dataset(path)
    print("end read")
    #d2v_model = train_doc2vec(all_data)
    d2v_model = doc2vec.Doc2Vec.load("d2v.model")
    
    print("end doc2vec")
    classifier = train_classifier(d2v_model, x_train, y_train)
    print("end train")
    test_classifier(d2v_model, classifier, x_test, y_test)
    print("end test")
