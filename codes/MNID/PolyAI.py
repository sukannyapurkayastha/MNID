#!/usr/bin/env python
# coding: utf-8





# In[4]:


import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tqdm.auto import tqdm

module_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/5'
# Import the Universal Sentence Encoder's TF Hub module
embed = hub.load(module_url)
print(embeddings)


# In[5]:


# embed text using Universal Sentence Encoder within 512 dimension size
# In[67]:
def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)))

# In[68]:
embed_size = 512


# In[6]:


df_train = pd.read_csv('dataset/train.csv',header=None, names=['text','label'])
print(len(df_train['label'].unique()))
num_classes = len(df_train['label'].unique()) #Safely assuming that train and test set have same number of classes


# In[7]:



y_train = df_train.label.values
encoder = LabelEncoder()
encoder.fit(y_train) # encoder is fit between number of classes in train

train_text = df_train['text'].tolist()
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
y_train = df_train.label.values
encoded_Y_train = encoder.transform(y_train)  # encode train labels to 0 to n-1 classes
train_label = to_categorical(encoded_Y_train) #inverse transform for encoded classes
gold_label_train = df_train['label'].tolist()

df_test = pd.read_csv('dataset/test.csv',header=None, names=['text','label'])
print("Train Classes: ", len(list(df_train['label'].unique())), "Test Classes: ", len(list(df_test['label'].unique())))


test_text = df_test['text'].tolist()
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
y_test = df_test.label.values
encoded_Y_test = encoder.transform(y_test) # encode test labels to 0 to n-1 classes
test_label = to_categorical(encoded_Y_test) #inverse transform for encoded classes
gold_label_test = df_test['label'].tolist()

print(df_test.shape[0])
print("test_text: ", len(test_text))


# In[29]:


# Implementation of the neural network
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7) #early-stopping
input_text = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
embedding = tf.keras.layers.Lambda(UniversalEmbedding, output_shape=(embed_size,))(input_text)
dense = tf.keras.layers.Dense(512, activation='relu')(embedding)
dropout = tf.keras.layers.Dropout(0.75)(dense)
pred = tf.keras.layers.Dense(num_classes, activation='softmax')(dropout)
model = tf.keras.Model(inputs=[input_text], outputs=pred)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.7,decay_steps=10000, decay_rate=0.9) #exponential decay
sgd = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()

history = model.fit(train_text, train_label, epochs=100, batch_size=15, validation_split=0.1, callbacks=[callback])


# In[30]:


# Performance evaluation
predicts = model.predict(test_text, batch_size=32)
train_predicts = model.predict(train_text,batch_size=32)

predict_logits = predicts.argmax(axis=1) # take the prediction at the index which has maximum value
predict_train_logits = train_predicts.argmax(axis=1)

(total,count)=(0,0)
predictions = predict_logits.tolist()
predictions = encoder.inverse_transform(predictions)
predictions_test = predictions
for gold,pred in zip(gold_label_test,predictions):
    total+=1
    if gold==pred:
        count+=1
print(f'Accuracy on test set is: {round((count/total) * 100, 2)}') # prints the accuracy
preds_list = predictions_test.tolist()
from sklearn.metrics import classification_report
print(classification_report(gold_label_test,preds_list)) #prints the classification report







