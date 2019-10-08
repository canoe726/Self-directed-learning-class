import argparse
import sys
import numpy as np
from sklearn import metrics
import tensorflow as tf
from sklearn.model_selection import train_test_split
import openpyxl

data = np.genfromtxt('./Result/test.csv',
                         delimiter=',', skip_header=0, usecols=(0, 1), dtype=None, encoding='utf-8')

# create our training data from the tweets
x = [x[1] for x in data]

# index all the sentiment labels
y = np.asarray([x[0] for x in data])

FLAGS = None

MAX_DOCUMENT_LENGTH = 30 # 문서의 최대 길이. 길이가 1~25에 몰려 있어서 이렇게 했다.

EMBEDDING_SIZE = 200 # 단어를 벡터로 바꿔주는 것이 (Word) embedding 이다. 이 embedding의 크기를 얼마로 해주는지 설정해준다. 
                    # 많지 않은 데이터의 경우 차원을 크게 해주면 오히려 공간에서 흩어지므로 좋지 않을 것 같다.
n_words = 0
MAX_LABEL = 19
WORDS_FEATURE = 'words'  # Name of the input words feature. feature naming에 사용되므로, 큰 의미가 있는 것 같지는 않다.(아마도...?)
learning_rate = 0.01
test_size = 0.2

labels = ['official', 'musician', 'comedian', 'actor', 'soccer',
          'basketball',  'Baseball', 'athlete', 'poet', 'soldier',
          'businessmen', 'scholar', 'player', 'moviedirector', 'progamer',
          'religionist', 'journalist', 'artist', 'author']

labels_ko = ['공무원', '음악인', '개그맨', '배우', '축구',
          '농구',  '야구', '운동선수', '시인', '군인',
          '기업인', '학자', '연주가', '영화감독', '프로게이머',
          '종교인', '언론인', '예술가', '작가']

# define accuracy matrix
accuracy_matrix = [[0]*MAX_LABEL for i in range(MAX_LABEL)]

def estimator_spec_for_softmax_classification(logits, labels, mode):
    "Returns EstimatorSpec instance for softmax classification."
    predicted_classes = tf.argmax(logits, 1)
        
    # case 1) inference mode
    # ModeKey가 PREDICT이면 inference mode이다. inference는 학습시킨 모델을 새로운 데이터에 적용시키는 것을 말한다.
    if mode == tf.estimator.ModeKeys.PREDICT: 

        return tf.estimator.EstimatorSpec(mode=mode, predictions={'class': predicted_classes, 'prob': tf.nn.softmax(logits)})
    
    # case 2) training mode
    onehot_labels = tf.one_hot(labels, MAX_LABEL, 1, 0) 
    print(onehot_labels)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)    
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    # case 3) evaluation mode
    eval_metric_ops = {'accuracy': tf.metrics.accuracy(labels=labels, predictions=predicted_classes)}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def rnn_model(features, labels, mode):
    """RNN model to predict from sequence of words to a class."""
    # 단어들의 index를 embeddings로 바꾼다. 
    # [n_words(단어 개수), EMBEDDING_SIZE]의 크기를 가지는 matrix를 만들고 순서를 나타내는 단어들의 index를 
    # [batch_size, sequence_length, EMBEDDING_SIZE]에 mapping 한다.
    
    # word_vectors와 word_list를 만든다.
    word_vectors = tf.contrib.layers.embed_sequence(features[WORDS_FEATURE], vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
    word_list = tf.unstack(word_vectors, axis=1)

    # embedding size와 같은 hidden size를 가지는 GUR cell을 만든다.
    cell = tf.contrib.rnn.GRUCell(EMBEDDING_SIZE)

    # MAX_DOCUMENT_LENGTH의 길이를 가지는 RNN을 만든다. 그리고 각 유닛에 input으로 word_list를 준다.
    # (output, state)쌍이 return된다.
    _, encoding = tf.contrib.rnn.static_rnn(cell, word_list, dtype=tf.float32)

    # 마지막 유닛의 state 값을 softmax classification의 feature로 넘겨준다.
    logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

    return estimator_spec_for_softmax_classification(logits=logits, labels=labels, mode=mode)

def main(unused_argv):
    global n_words
    
    # train data set, test data set을 나눠준다.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 4)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test) # tensorflow input으로 사용될 수 있는 자료형으로 변환시켜줘야 한다.
    
    # 단어들을 우리가 원하는 sequence length로 맞추어 준다.(embedding 해주기 전 단계) 위의 사진 참고
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
    
    x_transform_train = vocab_processor.fit_transform(x_train) # 둘의 차이는?
    x_transform_test = vocab_processor.transform(x_test)

    x_train = np.array(list(x_transform_train))
    x_test = np.array(list(x_transform_test))
    
    n_words = len(vocab_processor.vocabulary_)
    print('Total words : ', n_words)
    
    # 모델을 만들어준다.(여기서는 위에서 정의한 rnn_model을 사용한다.)
    model_fn = rnn_model
    classifier = tf.estimator.Estimator(model_fn=model_fn)
    
    # Train
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {WORDS_FEATURE : x_train},
        y = y_train,
        batch_size = len(x_train),
        num_epochs = None,
        shuffle = True) # shuffle = True : 
    classifier.train(input_fn = train_input_fn, steps = 100)
    
    # Predict
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {WORDS_FEATURE : x_test},
        y = y_test,
        num_epochs = 1,
        shuffle = False)
    predictions = classifier.predict(input_fn = test_input_fn)
    y_predicted = np.array(list(p['class'] for p in predictions))

    # print("y_test : ", y_test)
    # print("y_predicted : ",y_predicted)

    # Score using tensorflow
    score = classifier.evaluate(input_fn = test_input_fn)
    print('Accuracy (tensorflow): {0:f}'.format(score['accuracy']))
    
    for i in range(0, len(y_test - 1)):
        accuracy_matrix[ y_test[i] ][ y_predicted[i] ] += 1
    
    # print accuracy matrix
    correct = 0
    total_size = MAX_LABEL*100
    total_test_case = int(total_size * test_size)

    print("\ntotal_size : ", total_size)
    print("train_size : ", total_size - total_test_case)
    print("test_size  : ", total_test_case)

    print("\n-----Accuracy Matrix-----")
    for i in range(-1, MAX_LABEL) :
      for j in range(-1, MAX_LABEL) :
        
        if( i == j ) : correct += accuracy_matrix[i][j]
        
        if i == -1 :
          if j == -1 :
              print("C/P ",end='')
              continue
          if j == MAX_LABEL :
              print()
              break
          print(labels[j][:3]+" ", end='')
        
        else :
          if j == -1 :
              print(labels[i][:3], end='')
              continue
          if j == MAX_LABEL :
              print()
              break
          print("%4d"%accuracy_matrix[i][j], end='')
            
      print()  

    print("correct : ",correct)
    print('\nTotal Accuracy : ', correct / total_test_case)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_with_fake_data',
        default = False,
        help = 'Test the example code with fake data',
        action = 'store_true')
    parser.add_argument(
        '--bow model',
        default = False,
        help = 'Run with BOW model instead of RNN',
        action = 'store_true')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv = [sys.argv[0]] + unparsed)
