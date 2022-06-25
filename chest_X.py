import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#ECG CNN 모델

import sys
import tensorflow as tf
import keras
#from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pylab as plt
import cv2
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import random
#from keras import keras
from tensorflow.keras.layers import BatchNormalization

from sklearn.model_selection import KFold   # corss - validation 
from skimage.transform import resize
from PIL import Image
from skimage import exposure
from keras.preprocessing.image import ImageDataGenerator

#import matplotlib.pyplib as plt
#from skleran.metrics import classification_report,confusion_matrix
#from vis import plot_confusion_matrix as conf_mat
#------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

os.environ['TF_CPP_MIN_LOG_LEVEL'] ='3'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#gpus = tf.config.experimental.list_physical_devices('GPU')

#print("gpu:",gpus)

gpus = tf.config.experimental.list_physical_devices('GPU') 
if gpus: 
    try: 
        for gpu in gpus: 
            tf.config.experimental.set_memory_growth(gpu, True) 
        logical_gpus = tf.config.experimental.list_logical_devices('GPU') 
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs") 
    except RuntimeError as e: 
        print(e)
        
        
print(gpus)




datadir='X:/inteliigience x-ray/nih_gwang'
folderlist=os.listdir(datadir)




#print(tf.__version__) #텐서플로우 버젼 2.7.0
#label & one-hot encoding
label_encoder = LabelEncoder()
integer_encoded=label_encoder.fit_transform(folderlist)
onehot_encoder=OneHotEncoder(sparse=False) 
integer_encoded=integer_encoded.reshape(len(integer_encoded),1)
one=[]

num_classes =2
for t in range(0,num_classes):
    one.append(integer_encoded[t][0])
"""for i in range(0,15):
    one=np.stack(np.array(onehot_encoded[i]))"""





img_rows =150
img_cols =150
trains=[]
tests=[]


def minmax(data):
    data_mm=((data-np.min(data))/(np.max(data)-np.min(data)))
    #print('minmax')
    return data_mm#0~1로 정규화  

def histo(img):
    
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img = cdf[img]
    return img    
    
for i in range(len(folderlist)):
    imgdir=datadir+'/'+folderlist[i]
    imgdir2=os.listdir(imgdir)
    
    traindir=imgdir+'/'+imgdir2[1]
    testdir=imgdir+'/'+imgdir2[0]
    
    trainlist=os.listdir(traindir)
    testlist=os.listdir(testdir)
    
    for o in trainlist:
        imtr=cv2.imread(traindir+'/'+o,cv2.IMREAD_GRAYSCALE)
        
        #imtr=minmax(imtr)
        imtr=histo(imtr)
        
        
        imtrre = cv2.resize(imtr, (img_rows, img_cols))
        imtrre = imtrre.astype('float32')/255
        (train_x,train_y)=imtrre,one[i]
        trains.append(([np.array(train_x)],[np.array(train_y)]))
        
    for u in testlist:
        imtxt=cv2.imread(testdir+'/'+u,cv2.IMREAD_GRAYSCALE)
       
        #imtxt=minmax(imtr2)
        imtxt=histo(imtxt)
        
        imtxt = cv2.resize(imtxt, (img_rows, img_cols))
        imtxt = imtxt.astype('float32')/255
        
        (test_x,test_y)=imtxt,one[i]
        tests.append(([np.array(test_x)],[np.array(test_y)]))
            
    print("이미지 로드 완료")    
   
 
#각 train과 test tensor들 random 하게 shuffle     
random.shuffle(trains)
random.shuffle(tests)


#shuffle 후 다시 분리 x,y data끼리 분리
x_train,y_train=[],[]
x_test,y_test=[],[]

for i in range(0,len(trains)):
    x_train.append(np.array(trains[i][0]))
    y_train.append(np.array(trains[i][1]))
    
for i in range(0,len(tests)):
    x_test.append(np.array(tests[i][0]))
    y_test.append(np.array(tests[i][1]))

x_train=np.array(x_train)
y_train=np.array(y_train)
x_test=np.array(x_test)
y_test=np.array(y_test)


#COUNT_NORMAL=11880
#COUNT_PNEUMONIA=1080
TRAIN_IMG_COUNT = x_train.shape[0]


del(trains)
del(tests)
del(train_x)
del(test_x)
del(train_y)
del(test_y)

#%%
        

    
input_shape = (img_rows, img_cols, 1)  #color image가 아니므로 pixcel *pixcel*1
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)  #image 크기 변경
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#%%
batch_size =16
epochs =100
init=tf.keras.initializers.HeNormal#가중치 초기화 He
#%%

folder_splits=3
kfold = KFold(n_splits=folder_splits, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 0
accsum=[0] *epochs
losssum=[0] *epochs
accsum_val=[0] *epochs
losssum_val=[0] *epochs


"""
weight_for_0 = (1 / COUNT_NORMAL)*(TRAIN_IMG_COUNT)/2.0 
weight_for_1 = (1 / COUNT_PNEUMONIA)*(TRAIN_IMG_COUNT)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}
"""
#%%data augumentation
trdatagen = ImageDataGenerator(
    #featurewise_center=True,
    #zca_whitening=True,
    rotation_range=5,
                             width_shift_range=0.1, 
                             #shear_range=0.0,
                             zoom_range=0.1,
                             horizontal_flip=True,
                             vertical_flip=False,
                             fill_mode='nearest'
                            )






#%% weoght 추가

COUNT_NORMAL=1341
COUNT_PNEUMONIA=3875
TRAIN_IMG_COUNT = x_train.shape[0]


weight_for_0 = (1 / COUNT_NORMAL)*(TRAIN_IMG_COUNT)/2 
weight_for_1 = (1 / COUNT_PNEUMONIA)*(TRAIN_IMG_COUNT)/2

class_weight = {0: weight_for_0, 1: weight_for_1}



#%%model

model_name='gwang_NIH' 
modelnumber=0


for train_index, val_index in kfold.split(x_train,y_train):  #index를 받아옴
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3,3), strides=(1, 1), padding='valid', activation='relu', input_shape=input_shape,kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3,3), strides=(1, 1), padding='valid', activation='relu', input_shape=input_shape,kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    
    
    #22-05-22 주석처리
    model.add(Conv2D(256, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape,kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape,kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    
    model.add(Flatten())
    model.add(Dense(128, activation='relu',kernel_initializer=init)) #128->256   5.21
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu',kernel_initializer=init))
    model.add(Dropout(0.3))
    
    model.add(Dense(64, activation='relu',kernel_initializer=init))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=1e-6)
    with tf.device("/device:GPU:0"):
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
       
        
        #기본
        #hist = model.fit(x_train[train_index], y_train[train_index], batch_size=batch_size, epochs=epochs, verbose=1,validation_data=(x_train[val_index], y_train[val_index]),
                         #class_weight=class_weight  #validation 
      #)
        
        
        #dataaugumentation시
        hist = model.fit(trdatagen.flow(x=x_train[train_index],y=y_train[train_index],batch_size=32,shuffle=True), batch_size=batch_size,
                         epochs=epochs, verbose=1 #validation_data=val_datagen.flow(x=x_train[val_index], y=y_train[val_index],batch_size=32,shuffle=True)
                         
       #                  validation_data=(x_train[val_index], y_train[val_index])
                         #class_weight=class_weight  #validation 
                         )
        
    
    score = model.evaluate(x_test, y_test, verbose=0)
    testscore=[]
    testscore.append([score])
    
    
    for e in range(0,epochs):
        accsum[e]=accsum[e]+hist.history['accuracy'][e]
        losssum[e]=losssum[e]+hist.history['loss'][e]
        accsum_val[e]= accsum_val[e] + hist.history['val_accuracy'][e]
        losssum_val[e]=losssum_val[e]+hist.history['val_loss'][e]
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    fold_no+1
    
    # 6 훈련 과정 시각화 (정확도)
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])  # validation loss & accuracy
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train accuracy', 'validation accuracy'], loc='upper left')
    plt.show()
    # 7 훈련 과정 시각화 (손실)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train loss', 'validation loss'], loc='upper left')
    plt.show()
    
    
    
    modelnumber+=1
    model.save(model_name+str(modelnumber)+'.h5')

for t in range(0,epochs):
        
    accsum[t]/=folder_splits
    losssum[t]/=folder_splits
    accsum_val[t]/=folder_splits
    losssum_val[t]/=folder_splits
    
#print(accsum,losssum)
    
plt.plot(accsum)
plt.plot(accsum_val)  # validation loss & accuracy
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train accuracy', 'validation accuracy'], loc='upper left')
plt.show()
# 7 훈련 과정 시각화 (손실)
plt.plot(losssum)
plt.plot(losssum_val)
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train loss', 'validation loss'], loc='upper left')
plt.show()
