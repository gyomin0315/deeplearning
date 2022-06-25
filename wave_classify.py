import os
os.add_dll_directory('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.6\\bin')


import pandas as pd

data = pd.read_csv('wave_dataset_pow_10.csv')

test = pd.read_csv('test5_10_sin2x_pow_10.csv')


print(data)

##y데이터 = data['wave'].values

y데이터 = []

x데이터 = []

test_data = list(test)





for i, rows in data.iterrows():
    y데이터.append([rows['sin'], rows['cos'], rows['sin2x'], rows['cos2x'], rows['3sinx'], rows['3cosx']])
    


for i, rows in data.iterrows():
    x데이터.append([rows['x1'], rows['x2'], rows['x3'],rows['x4'], rows['x5'], rows['x6'], rows['x7'], rows['x8'], rows['x9'],rows['x10'], rows['x11'], rows['x12'],rows['x13'], rows['x14'], rows['x15'],rows['x16'], rows['x17'], rows['x18'], rows['x19'], rows['x20'], rows['x21'],rows['x22'], rows['x23'], rows['x24'], rows['x25'], rows['x26'],rows['x27'], rows['x28'], rows['x29'], rows['x30'], rows['x31'],rows['x32'], rows['x33'], rows['x34'], rows['x35'], rows['x36'],rows['x37'], rows['x38'], rows['x39'], rows['x40'], rows['x41'],rows['x42'], rows['x43'], rows['x44'], rows['x45'], rows['x46'],rows['x47'], rows['x48'], rows['x49'], rows['x50'], rows['x51'],rows['x52'], rows['x53'], rows['x54'], rows['x55'], rows['x56'],rows['x57'], rows['x58'], rows['x59'], rows['x60'], rows['x61'],rows['x62'], rows['x63']])


import numpy as np
import tensorflow as tf

#np.set_printoptions(precision=3)

model = tf.keras.models.Sequential([    
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax'),                    
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(np.array(x데이터), np.array(y데이터), epochs=300 )

#model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

#예측
#예측값 = model.predict([ [-0.0576990264230644,	0.0818434527066484,	0.317697172401646,	0.237967796589291,	0.465008192765241,	0.331620999190504,	0.554356306305263,	0.697059432903586,	0.917380394885728,	0.682084481461216,	0.893800541867596,	0.778205155274993,	1.02051591589761,	1.05678691292748,	1.14512614387386,	0.905574859392256,	0.915767054994471,	0.848440190318510,	1.06995725241648,	1.04460517369829,	0.952197277341920,	0.791229283066468,	0.905487686502991,	0.775807903347035,	0.878491114118076,	0.487373801327712,	0.476951245505182,	0.462573769978788,	0.185136947971604,	0.121786406952794,	0.272067985533875,	-0.111530249077598,	-0.0400644029902372,	-0.0666601588938737,	-0.0236072969011922,	-0.462091537305737,	-0.360250552869644,	-0.532943729609373,	-0.343184388147021,	-0.696158005315445,	-0.732058075023624,	-0.737558296315426,	-0.701184695352465,	-0.939902088380448,	-1.00392786108070,	-0.850978789950912,	-0.925752594521175,	-0.884997688489346,	-0.880771552473086,	-0.930298337980589,	-0.785806781890299,	-0.965829629947730,	-0.915302738555370,	-0.906973564628384,	-0.668077657518157,	-0.483071020094844,	-0.748465802247575,	-0.506400505402395,	-0.411069466561625,	-0.123646622429812,	-0.432137400675493,	-0.265756159553191,	-0.0771481261149627] ])
#예측값 = model.predict([[0.119694587247009,	0.124175785722639,	0.608594060243966,	0.574008312887214	,0.784793841659397,	0.868216207342574,	0.619943807393868,	0.931957482583339,	0.930316272473108	,0.687209841762636,	0.812601706994139,	0.825685931822620,	0.637333982134121,	0.466565635618556,	0.315838504577312,	0.132658794190626,	-0.171934228576956,	-0.401230937924807,	-0.363704429847092,	-0.596121390292360,	-0.653284075941097,	-0.854074582153258,	-0.961493942381364,	-1.00890514907573	,-0.853063729572768,	-0.867517831814078,	-0.763111324311777,	-0.727071800030447,	-0.720148262146301,	-0.555803753968385,	-0.294882902118802,	-0.0295033073452222,	0.0931293402198776,	0.362457843809061	,0.474395687854935,	0.682091492356834,	0.757216636657643,	0.914366349523236,	1.07569246678792,	0.843501458149211,	1.01972443952760,	1.03523662618097,	0.866865384633179	,0.607890071969596,	0.406009902923206,	0.389214860089861,	0.178671303507355,	-0.0463222576548274,	-0.191486777931812,	-0.451770004158867,	-0.473108357206757,	-0.557573004233261,	-0.763590245807251,	-0.972566202769226,	-0.989689496054988,	-0.948667047641443,	-0.965162382156789,	-1.18762714830110	,-0.788568356244054,	-0.675391758061362,	-0.418952612051434,	-0.525127156586881,	-0.230604772124829]]) #sin2x
#예측값 = model.predict([[0.882706807396946,	1.03577325873701,	0.956432884345896,	1.09383734335073,	0.958519792265746,	0.793124014708590,	0.923175184790515,	0.885364702070165,	0.722489636084391,	0.632751307039135,	0.530055297617831, 0.256192325071236,	0.347814241165385,	0.315292135581171,	0.129494403966801,	-0.00119621363221586,	-0.152449157335149,	-0.255184190232156,	-0.337392643136652,	-0.401609040809994,	-0.383692041581461,	-0.295583236267737,	-0.451882817832123,	-0.522488723268020,	-0.616396118798545,	-0.720915926785424,	-0.735788395698433,	-0.945709393070023,	-0.966539794940356,	-0.879375987665079,	-1.03005617719186, -0.928996343864365,	-0.916164921830666,	-1.00100437443035,	-0.887616189634584	, -1.03442988464768,	-0.966360643749279,	-1.02407637826831, -0.768056958704971,	-0.763063816354570,	-0.593352168486341,	-0.611839293329012,	-0.517734153783993,	-0.323745087137306,	-0.172872182610155,	-0.124776403881549,	-0.173488667232918,	0.0330059670010255,	0.0500299077600829,	0.140668144012814,	0.229029996810504, 0.352854078916143,	0.599407478633691,	0.517369479211320,	0.597692525694610,	0.789147328647996,	0.866677453556334,	0.815215410448675,	0.869486847534310,	1.04232161510101,	1.01364207083733,	0.966385879484125,	0.961848066363709]])

#예측값 = model.predict([[-0.782339460234994,	-5.48772096925141,	-0.395339696068531,	-1.00002610412676, 0.974368172270643,	8.11499372178809,	0.572613667149759,	1.83078860908165,	2.66291520513886,	-7.18963546817887,	1.40191051891293,	1.60409908114130,	0.832454256196152,	9.26875407706507,	1.73984755479447,	1.02042627445936,	0.518029336691234,	-1.58340986547141,	1.19533895846448,	0.227011882135941,	0.282727286886515,	-7.98127738472912,	-1.79364417536721,	-1.49325093928301,	-3.21049181955499,	6.113359998376366,	-0.854114254265943,	-1.05215784653050,	-6.952219387101838,	0.0160014835126706,	5.465386390635774,	-1.77060750197522,	3.936930397306263,	-0.101927726104564,	-0.421108433724996,	0.335767862674671,	0.0242303434061033,	-2.496893055462254,	1.87950060416360,	1.03529005694531,	-5.42573312718396,	1.24713777088156,	1.40044531112816,	2.60894055782524,	-2.14022845906481,	0.538237628579237,	2.67730342591339,	-5.31359763934881,	0.137217943571556,	0.362848697569974,	0.457359448039132,	0.831571459794025,	-2.40419315390830,	-5.39859211405717,	-0.309355849523013,	-2.06897053770662,	0.645398866354301,	-0.665737410205513,	-5.46263053033004,	-1.46386946673822,	-0.949673705124102,	-0.937968010992282,	0.575775553639839]]) #sin2x with 10x whitenoise

##for i, 예측값 in enumerate

예측값 = model.predict([list(np.float_(test_data))])

예측값 = 예측값 * 100

#예측값 = 예측값.astype(int)

print('sinx:',format(예측값[0][0],'.5f'),'%')
print('cosx:',format(예측값[0][1],'.5f'),'%')
print('sin2x:',format(예측값[0][2],'.5f'),'%')
print('cos2x:',format(예측값[0][3],'.5f'),'%')
print('3sinx:',format(예측값[0][4],'.5f'),'%')
print('3cosx:',format(예측값[0][5],'.5f'),'%')


