import os
os.add_dll_directory('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.6\\bin')


import pandas as pd

data = pd.read_csv('wave_dataset_pow_7_8.csv')

test_1 = pd.read_csv('test5_pow_8_sinx.csv')
test_2 = pd.read_csv('test5_pow_8_cosx.csv')
test_3 = pd.read_csv('test5_pow_8_sin2x.csv')
test_4 = pd.read_csv('test5_pow_8_cos2x.csv')
test_5 = pd.read_csv('test5_pow_8_3sinx.csv')
test_6 = pd.read_csv('test5_pow_8_3cosx.csv')
test_7 = pd.read_csv('test5_pow_7_sinx.csv')
test_8 = pd.read_csv('test5_pow_7_cosx.csv')
test_9 = pd.read_csv('test5_pow_7_sin2x.csv')
test_10 = pd.read_csv('test5_pow_7_cos2x.csv')
test_11 = pd.read_csv('test5_pow_7_3sinx.csv')
test_12 = pd.read_csv('test5_pow_7_3cosx.csv')

print(data)

y데이터 = []

x데이터 = []

test_data_1 = list(test_1)
test_data_2 = list(test_2)
test_data_3 = list(test_3)
test_data_4 = list(test_4)
test_data_5 = list(test_5)
test_data_6 = list(test_6)
test_data_7 = list(test_7)
test_data_8 = list(test_8)
test_data_9 = list(test_9)
test_data_10 = list(test_10)
test_data_11 = list(test_11)
test_data_12 = list(test_12)


#print(test_data_1)


for i, rows in data.iterrows():
    y데이터.append([rows['sin'], rows['cos'], rows['sin2x'], rows['cos2x'], rows['3sinx'], rows['3cosx'],rows['sin_7'], rows['cos_7'], rows['sin2x_7'], rows['cos2x_7'], rows['3sinx_7'], rows['3cosx_7']])


for i, rows in data.iterrows():
    x데이터.append([rows['x1'], rows['x2'], rows['x3'],rows['x4'], rows['x5'], rows['x6'], rows['x7'], rows['x8'], rows['x9'],rows['x10'], rows['x11'], rows['x12'],rows['x13'], rows['x14'], rows['x15'],rows['x16'], rows['x17'], rows['x18'], rows['x19'], rows['x20'], rows['x21'],rows['x22'], rows['x23'], rows['x24'], rows['x25'], rows['x26'],rows['x27'], rows['x28'], rows['x29'], rows['x30'], rows['x31'],rows['x32'], rows['x33'], rows['x34'], rows['x35'], rows['x36'],rows['x37'], rows['x38'], rows['x39'], rows['x40'], rows['x41'],rows['x42'], rows['x43'], rows['x44'], rows['x45'], rows['x46'],rows['x47'], rows['x48'], rows['x49'], rows['x50'], rows['x51'],rows['x52'], rows['x53'], rows['x54'], rows['x55'], rows['x56'],rows['x57'], rows['x58'], rows['x59'], rows['x60'], rows['x61'],rows['x62'], rows['x63']])


import numpy as np
import tensorflow as tf



model = tf.keras.models.Sequential([    
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(12, activation='softmax'),                    
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(np.array(x데이터), np.array(y데이터), epochs=800 )


예측값 = model.predict([list(np.float_(test_data_1)),list(np.float_(test_data_2)),list(np.float_(test_data_3)),list(np.float_(test_data_4)),list(np.float_(test_data_5)),list(np.float_(test_data_6)),list(np.float_(test_data_7)),list(np.float_(test_data_8)),list(np.float_(test_data_9)),list(np.float_(test_data_10)),list(np.float_(test_data_11)),list(np.float_(test_data_12))])

예측값 = 예측값 * 100

for i in range(len(예측값)):
    print()
    print('test_',i+1)
    print()
    print('sinx:',format(예측값[i][0],'.5f'),'%')
    print('cosx:',format(예측값[i][1],'.5f'),'%')
    print('sin2x:',format(예측값[i][2],'.5f'),'%')
    print('cos2x:',format(예측값[i][3],'.5f'),'%')
    print('3sinx:',format(예측값[i][4],'.5f'),'%')
    print('3cosx:',format(예측값[i][5],'.5f'),'%')
    print('sinx_7:',format(예측값[i][6],'.5f'),'%')
    print('cosx_7:',format(예측값[i][7],'.5f'),'%')
    print('sin2x_7:',format(예측값[i][8],'.5f'),'%')
    print('cos2x_7:',format(예측값[i][9],'.5f'),'%')
    print('3sinx_7:',format(예측값[i][10],'.5f'),'%')
    print('3cosx_7:',format(예측값[i][11],'.5f'),'%')
    print()