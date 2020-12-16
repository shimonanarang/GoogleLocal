# imports
import pandas as pd
import pickle
import sklearn
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import gensim
from gensim.models import Word2Vec, KeyedVectors
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import sklearn
from sklearn.preprocessing import OneHotEncoder
import tensorflow.keras as keras
from tensorflow.keras.layers import Concatenate, Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D,MaxPooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Convolution1D as Conv1D
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.metrics import roc_curve, auc, confusion_matrix,accuracy_score,precision_score
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import metrics
from itertools import cycle
from scipy import interp
from sklearn.metrics import roc_curve, auc

# reading dataset
dset = pd.read_pickle('restaurants_places.pickle')

# removing all 0 ratings from dataset
dset = dset[dset['rating']!=0]

# loading word2vec model
vec_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=300000) # each vector is of length 300

# convert reviews in dataset set into list of word vectors
train_vector = []
for idx, row in dset.iterrows():
    sentence_vec = []
    sentence = row['reviewText']
    sentence = word_tokenize(sentence)
    for word in sentence:
        try:
            sentence_vec.append(vec_model[word])
        except Exception as e:
            pass
    train_vector.append(sentence_vec)

# trimm reviews to be of equal length, pad shorter reviews with 0
# pad zeros to the vectors to make them of equal length
zero_pad = np.zeros((300, ))
trim_length = 40
trimmed_train_vector = []
for item in train_vector:
    if len(item)>trim_length:
        trimmed_train_vector.append(item[:trim_length])
    else:
        len_zero_pads = trim_length - len(item)
        for i in range(len_zero_pads):
            item.append(zero_pad)
        trimmed_train_vector.append(item)


# converting trimmed data into np array
trimmed_train_vector = np.array(trimmed_train_vector, dtype = np.float32)

ratings = np.array(dset['rating'])
enc = OneHotEncoder()
# one-hot encode the train and test ratings 
ratings = enc.fit_transform(ratings.reshape(-1, 1)).toarray()

# split the data into training and test set
split = 0.9
train_len = int(split * len(ratings))
train_reviews = trimmed_train_vector[:train_len]
train_ratings = ratings[:train_len]
test_reviews = trimmed_train_vector[train_len:]
test_ratings = ratings[train_len:]

# model architecture
dropout_p = 0.3
avg_len = trim_length

embed_in = Input(shape=(trim_length, 300))
parallel_conv = []

#1st
conv = Conv1D(256,kernel_size= 2, kernel_regularizer=regularizers.l2(l=0.01),
              kernel_initializer='he_normal',padding="valid", activation="relu", strides=1)(embed_in)
conv = Dropout(dropout_p)(conv)
pool_conv = MaxPooling1D(pool_size = avg_len - 2 + 1)(conv)
flatten_conv = Flatten()(pool_conv)
parallel_conv.append(flatten_conv)

#2nd
conv = Conv1D(256,kernel_size= 3,kernel_regularizer=regularizers.l2(l=0.01), padding="valid", activation="relu", strides=1)(embed_in)
conv = Dropout(dropout_p)(conv)
pool_conv = MaxPooling1D(pool_size = avg_len - 3 + 1)(conv)
flatten_conv = Flatten()(pool_conv)
parallel_conv.append(flatten_conv)

#3rd
conv = Conv1D(256,kernel_size= 4, kernel_regularizer=regularizers.l2(l=0.01),padding="valid", activation="relu", strides=1)(embed_in)
conv = Dropout(dropout_p)(conv)
pool_conv = MaxPooling1D(pool_size = avg_len - 4 + 1)(conv)
flatten_conv = Flatten()(pool_conv)
parallel_conv.append(flatten_conv)

#4th
conv = Conv1D(256,kernel_size= 5, kernel_regularizer=regularizers.l2(l=0.01),padding="valid", activation="relu", strides=1)(embed_in)
conv = Dropout(dropout_p)(conv)
pool_conv = MaxPooling1D(pool_size = avg_len - 5 + 1)(conv)
flatten_conv = Flatten()(pool_conv)
parallel_conv.append(flatten_conv)

#concatenate the parallel Convulations
out = Concatenate()(parallel_conv)

graph = Model(inputs=embed_in, outputs=out, name="parallelConv")
graph.summary()

model = Sequential()
model.add(graph)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(ratings.shape[1], activation='sigmoid'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", metrics.AUC()])
model.summary()

label_counts = train_ratings.sum(axis = 0)
train_size = len(train_ratings)

class_weight = {0: train_size / ( 5 * label_counts[0]),  #rating 1
                1: train_size / ( 5 * label_counts[1]),
                2: train_size / ( 5 * label_counts[2]),
                3: train_size / ( 5 * label_counts[3]),
                4: train_size / ( 5 * label_counts[4])}  #rating 5

history = model.fit(train_reviews, train_ratings, validation_split=0.2, epochs=4,batch_size=128, class_weight=class_weight)

# plot the loss curve for training and validation set
plt.plot(history.history['loss'],label = "Train Loss")
plt.plot(history.history['val_loss'],label = 'Validation Loss')
plt.legend()
plt.show()

# evaluate on test set, get loss, accuracy and auc on test set
score, acc, auc = model.evaluate(test_reviews, test_ratings)

# construct confusion matrix
prob_test = np.round(model.predict(test_reviews))
prob_test = pd.DataFrame(data = prob_test, columns = ['1', '2', '3','4','5'])
prob_test = prob_test.idxmax(axis = 1)

test_ratings_df = pd.DataFrame(data = test_ratings, columns = ['1', '2', '3','4','5'])
y_test_label = test_ratings_df.idxmax(axis = 1)

# Calculate confusion matrix
confusion_matrix_dnn = confusion_matrix(y_true = y_test_label, 
                    y_pred = prob_test)

# Turn matrix to percentages
confusion_matrix_dnn = confusion_matrix_dnn.astype('float') / confusion_matrix_dnn.sum(axis=1)[:, np.newaxis]

# Turn to dataframe
df_cm = pd.DataFrame(
        confusion_matrix_dnn, index=['1', '2', '3','4','5'],
         columns=['1', '2', '3','4','5'], 
)

# Parameters of the image
figsize = (10,7)
fontsize=14

# Create image
fig = plt.figure(figsize=figsize)
heatmap = sns.heatmap(df_cm, annot=True, fmt='.2f')

# Make it nicer
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, 
                             ha='right', fontsize=fontsize)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45,
                             ha='right', fontsize=fontsize)

# Add labels
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Plot!
plt.show()

# plot AUC-ROC curve
preds = model.predict(test_reviews)
n_classes = 5
lw =2

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_ratings[:, i], preds[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i+1, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
