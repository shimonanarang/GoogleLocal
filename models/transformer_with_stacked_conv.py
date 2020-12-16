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
from sklearn.preprocessing import OneHotEncoder
import gensim
from gensim.models import Word2Vec, KeyedVectors
from nltk.tokenize import sent_tokenize, word_tokenize
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
from tensorflow.keras.callbacks import ModelCheckpoint

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

# define multi-head attention block
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output

# define transformer block
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.3):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# define model
embed_dim = 32  # Embedding size for each token
num_heads = 8  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(50, 300))
transformer_block = TransformerBlock(300, 10, 150)
x = transformer_block(inputs)
x = Conv1D(256,kernel_size= 3, kernel_regularizer=regularizers.l2(l=0.01),padding="valid", activation="relu", strides=1)(x)
x = layers.MaxPooling1D(pool_size=2, strides=None, padding="valid")(x)
x = Conv1D(128,kernel_size= 3, kernel_regularizer=regularizers.l2(l=0.01),padding="valid", activation="relu", strides=1)(x)
x = layers.MaxPooling1D(pool_size=2, strides=None, padding="valid")(x)
x = Conv1D(64,kernel_size= 3, kernel_regularizer=regularizers.l2(l=0.01),padding="valid", activation="relu", strides=1)(x)
x = layers.MaxPooling1D(pool_size=2, strides=None, padding="valid")(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(5, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile("adam", "categorical_crossentropy", metrics=["accuracy", metrics.AUC()])
model.summary()

# fit model
history = model.fit(train_reviews, train_ratings, batch_size=128, epochs=10, validation_split=0.2)

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
