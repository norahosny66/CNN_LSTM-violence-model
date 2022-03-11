#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__  import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


import os


# In[5]:


fights_train = np.zeros((700, 40, 160, 160, 3), dtype=np.float)
labels_train = []


# In[3]:


def capture(filename):
    frames = np.zeros((40, 160, 160, 3), dtype=np.float)
    i=0
    vc = cv2.VideoCapture(filename)
    if vc.isOpened():
        rval , frame = vc.read()
    else:
        rval = False
    #frm = cv2.resize(frame,(200,200))
    frm = resize(frame,(160, 160, 3))
    frm = np.expand_dims(frm,axis=0)
    if(np.max(frm)>1):
        frm = frm/255.0
    frames[i][:] = frm
    i +=1
    while i < 40:
        rval, frame = vc.read()
        #print(i)
        #plt.imshow(frame)
        #plt.show()
        #frm = cv2.resize(frame,(200,200))
        frm = resize(frame,(160, 160, 3))
        frm = np.expand_dims(frm,axis=0)
        if(np.max(frm)>1):
            frm = frm/255.0
        frames[i][:] = frm
        i +=1
        #print(frame)
    return frames

def cut_save(main_dir,mod):
    i = 0
    #fights = np.zeros((399, 40, 200, 200, 3), dtype=np.float)
    #noFights = np.zeros((599, 42, 200, 200, 3), dtype=np.float)
    for x in os.listdir(main_dir):
        if 1 == 1:
            td = main_dir+x+'/'
            #for y in os.listdir(main_dir+x+'/'):
                #print(y)
            for file in os.listdir(td):
                fl = os.path.join(td, file)
                videos = capture(fl)
                if mod == 'train':
                    fights_train[i][:][:] = videos
                    i +=1
                    if x =='fights':
                        labels_train.append(1)
                    else:
                        labels_train.append(0)
                elif mod =='test':
                    fights_test[i][:][:] = videos
                    i +=1
                    if x =='fights':
                        labels_test.append(1)
                    else:
                        labels_test.append(0)
                elif mod =='val':
                    fights_val[i][:][:] = videos
                    i +=1
                    if x =='fights':
                        labels_val.append(1)
                    else:
                        labels_val.append(0)


# In[7]:


cut_save('./trainm/',"train")


# In[8]:


fights_train.shape


# In[4]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, fights_test, y_train, labels_test = train_test_split(fights_train,labels_train, test_size=0.33, random_state=42)


# In[ ]:


fights_train = []


# In[6]:


fights_test= np.zeros((300, 40, 160, 160, 3), dtype=np.float)
labels_test = []


# In[7]:


cut_save('./testm/',"test")


# In[8]:


plt.imshow(fights_test[19][5])
plt.show()


# In[10]:


layers = tf.keras.layers
models = tf.keras.models
losses = tf.keras.losses
optimizers = tf.keras.optimizers 
metrics = tf.keras.metrics
utils = tf.keras.utils
callbacks = tf.keras.callbacks
layers = tf.keras.layers
models = tf.keras.models
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
losses = tf.keras.losses
optimizers = tf.keras.optimizers 
metrics = tf.keras.metrics
utils = tf.keras.utils
callbacks = tf.keras.callbacks

plot_model = tf.keras.utils.plot_model


# In[11]:


np.random.seed(1234)
num_classes = 2


# In[12]:


np.random.seed(1234)
num_classes = 2
vg19 = tf.keras.applications.vgg19.VGG19
base_model = vg19(include_top=False,weights='imagenet',input_shape=(160, 160,3))
# Freeze the layers except the last 4 layers
for layer in base_model.layers[:-4]:
    layer.trainable = False
# Check the trainable status of the individual layers
base_model.summary()


# In[13]:


num_classes = 2

cnn = models.Sequential()
cnn.add(base_model)
cnn.add(layers.Flatten())
#cnn.add(layers.Dense(1024, activation='relu'))
#cnn.add(layers.Dropout(0.3))
#cnn.add(layers.Dense(512, activation='relu'))
#cnn.add(layers.Dropout(0.3))
#cnn.add(layers.LSTM(40))

# define LSTM model
model = models.Sequential()

model.add(layers.TimeDistributed(cnn,  input_shape=(40, 160, 160, 3)))
model.add(layers.LSTM(40 , return_sequences=True))

#model.add(layers.Dense(num_classes, activation="sigmoid"))
#model.add(layers.Dropout(0.3))

model.add(layers.TimeDistributed(layers.Dense(160, activation='relu')))

model.add(layers.GlobalAveragePooling1D(name="globale"))

'''
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.3))
'''
model.add(layers.Dense(num_classes, activation="sigmoid" , name="last"))

adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.load_weights('mamon98777.hdf5')
rms = optimizers.RMSprop()
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])
model.summary()


# In[32]:



class AccuracyHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_acc = []
        self.loss = []
        self.val_loss = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

history = AccuracyHistory()
earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=8,min_delta=1e-5, verbose=0, mode='min')
mcp_save = callbacks.ModelCheckpoint('mamon98777.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = callbacks.ReduceLROnPlateau(monitor='val_loss',patience=1, verbose=2,factor=0.5,min_lr=0.0000001)


# In[33]:


batch_size =3
epochs = 10


# In[34]:


y_train = utils.to_categorical(labels_train)
print(y_train)


# In[14]:


y_test = utils.to_categorical(labels_test)
print(y_test)


# In[36]:


import time
millis = int(round(time.time() * 1000))
print("started at " , millis)

model.fit(fights_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(fights_test, y_test),callbacks=[earlyStopping, mcp_save, reduce_lr_loss,history])

#0.8995 4


# In[19]:


fights_test = []


# In[37]:


acc = history.acc
val_acc = history.val_acc
loss = history.loss
val_loss = history.val_loss
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
 
plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
 
plt.show()


# In[20]:


fights_train = []


# In[15]:


score = model.evaluate(fights_test, y_test, batch_size=3)
score


# In[16]:


from sklearn.metrics import classification_report, confusion_matrix


# In[17]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


# In[18]:


Y_pred = model.predict(fights_test , batch_size=1)


# In[19]:


yprd = Y_pred > 0.5
yprd


# In[20]:


ypredicted = []
for zero,one in yprd:
    if zero == True:
        ypredicted.append(0)
    else:
        ypredicted.append(1) 


# In[21]:


ypredicted


# In[23]:


y_test


# In[24]:


y = []

for zero,one in y_test:
    if zero == True:
        y.append(0)
    else:
        y.append(1) 


# In[25]:


confusion = confusion_matrix(y,ypredicted)
confusion.shape


# In[26]:


print_confusion_matrix(confusion, [0,1], figsize = (30,15), fontsize=16)


# In[27]:


print('Classification Report')
print(classification_report(y, ypredicted, target_names=['no-violance','violance']))


# In[39]:


model.save("mamonbest980hocky.hdfs")

