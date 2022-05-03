#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tensorflow')
get_ipython().system('pip install gdown')
get_ipython().system('pip install tqdm')


# In[ ]:


import tensorflow as tf
import tensorflow_hub as hub
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tqdm.auto import tqdm

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"

embed = hub.load(module_url)


# In[ ]:


# In[67]:
def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)))

# In[68]:
embed_size = 512


# In[ ]:


df_train = pd.read_csv('data_generated/train.csv',header=None, names=['text','label'])
print(len(df_train['label'].unique()))
num_classes = len(df_train['label'].unique())


# In[ ]:


y_train = df_train.label.values
encoder = LabelEncoder()
encoder.fit(y_train)

train_text = df_train['text'].tolist()
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
y_train = df_train.label.values
encoded_Y_train = encoder.transform(y_train)
train_label = to_categorical(encoded_Y_train)
gold_label_train = df_train['label'].tolist()


# In[ ]:


input_text = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
embedding = tf.keras.layers.Lambda(UniversalEmbedding, output_shape=(embed_size,))(input_text)
dense = tf.keras.layers.Dense(512, activation='relu')(embedding)
dropout = tf.keras.layers.Dropout(0.75)(dense)
pred = tf.keras.layers.Dense(num_classes, activation='softmax')(dropout)
model = tf.keras.Model(inputs=[input_text], outputs=pred)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.7,decay_steps=10000,decay_rate=0.9)
sgd = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()

history = model.fit(train_text, train_label, epochs=100, batch_size=32)
train_predicts = model.predict(train_text,batch_size=32)
predict_train_logits = train_predicts.argmax(axis=1)
(total,count)=(0,0)
predictions = predict_train_logits.tolist()
predictions = encoder.inverse_transform(predictions)
for gold,pred in zip(gold_label_train,predictions):
    if gold==pred:
        count+=1
    total+=1
print(f'Accuracy on train set is:{round((count/total) * 100, 2)}') 


# In[ ]:


def rchop(s, suffix):
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    return s


# In[ ]:


directory = "preds"
  
# Parent Directory path
parent_dir = "data_generated"
  
# Path
path = os.path.join(parent_dir, directory)
os.mkdir(path)


# In[ ]:


def write_to_file(input_f, max_preds):
    f = open(f'data_generated/preds/{input_f}','w')
    for pred in max_preds:
        f.write(str(pred)+'\n')
    f.close()
    return 1


# In[ ]:


num_examples = [0]
files = []
df_test = pd.DataFrame()
for file in os.listdir("data_generated"):
    if file.startswith("gold_test"):
        print(file)
        test_file = os.path.join("data_generated",file)
        df_ = pd.read_csv(test_file,header=None, names=['text','label'])
        df_ = df_[df_['label'] != num_classes]
        df_test = pd.concat([df_test,df_])
        files.append(file)
        num_examples.append(df_test.shape[0])


# In[ ]:


print("Number of files: ", len(num_examples)-1, " == ", len(files))
print("Last num_example: ", num_examples[-1])

test_text = df_test['text'].tolist()
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
y_test = df_test.label.values
encoded_Y_test = encoder.transform(y_test)
test_label = to_categorical(encoded_Y_test)
gold_label_test = df_test['label'].tolist()

print(df_test.shape[0])
print("test_text: ", len(test_text))
print("Train Classes: ", len(list(df_train['label'].unique())), "Test Classes: ", len(list(df_test['label'].unique())))
predicts = model.predict(test_text, batch_size=32)
print(predicts.shape)
predict_logits = predicts.argmax(axis=1)

(total,count)=(0,0)
predictions = predict_logits.tolist()
predictions = encoder.inverse_transform(predictions)
for gold,pred in zip(gold_label_test,predictions):
    total+=1
    if gold==pred:
        count+=1
print(f'Accuracy on all test sets is:{round((count/total) * 100, 2)}')

max_preds = (predicts.max(axis=1)).tolist()

for i in range(0, len(num_examples)-1):
    file_ = rchop(files[i],'.csv')
    file_strip=file_.strip('.txt')
    u, v = num_examples[i], num_examples[i+1]
    temp = write_to_file(f'{file_strip}.txt', max_preds[u:v])
    if(temp != 1):
        print("File Writing not done")
        break


# In[ ]:


get_ipython().system("zip -r output_confidences/preds_new.zip 'data_generated/preds/'")


# In[ ]:




