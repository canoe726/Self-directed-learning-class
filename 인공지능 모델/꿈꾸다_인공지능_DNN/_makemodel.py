import json
import keras
import keras.preprocessing.text as kpt
import numpy as np

from keras import optimizers
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

class_num = 19

# extract data from a csv
# notice the cool options to skip lines at the beginning
# and to only take data from certain columns
training = np.genfromtxt('./Result/test.csv',
                         delimiter=',', skip_header=0, usecols=(0, 1), dtype=None, encoding='utf-8')

# create our training data from the tweets
train_x = [x[1] for x in training]

# index all the sentiment labels
train_y = np.asarray([x[0] for x in training])

# only work with the 5000 most popular words found in our dataset
max_words = 5000

# create a new Tokenizer
tokenizer = Tokenizer(num_words=max_words)

# feed our tweets to the Tokenizer
tokenizer.fit_on_texts(train_x)

# Tokenizers come with a convenient list of words and IDs
dictionary = tokenizer.word_index

# Let's save this out so we can use it later
with open('./Result/dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)

def convert_text_to_index_array(text):
    # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

allWordIndices = []

# for each tweet, change each token to its ID in the Tokenizer's word_index
for text in train_x:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

# now we have a list of all tweets converted to index arrays.
# cast as an array for future usage.
allWordIndices = np.asarray(allWordIndices)

# create one-hot matrices out of the indexed tweets
train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')

# treat the labels as categories
train_y = keras.utils.to_categorical(train_y, class_num)


model = Sequential()

model.add(Dense(1024, input_shape=(max_words,), activation='sigmoid'))
model.add(Dropout(0.5))

model.add(Dense(512, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, input_shape=(max_words,), activation='sigmoid'))
model.add(Dropout(0.5))

model.add(Dense(class_num, activation='softmax'))


model.compile(loss='categorical_crossentropy',
    optimizer='Adam',
    metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=256, epochs=100,
          
          verbose=1, validation_split=0.1, shuffle=True)

model_json = model.to_json()

with open('./Result/model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('./Result/model.h5')

print('saved model!')
