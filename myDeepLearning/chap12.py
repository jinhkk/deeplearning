# 12장 다중 분류 문제의 해결

import pandas as pd

df = pd.read_csv('./data/iris3.csv')

print(df.head())

# 그래프 확인
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df, hue='species')
'''
.pairplot(data, hue=기준) (위에서는 species[품종]을 기준으로 함)
'''
plt.show()

print('=' * 50)

# one-hot encoding

x = df.iloc[:,0:4]
y = df.iloc[:,4]

print(x[0:5])
print(y[0:5])

# one-hot encoding 처리

y = pd.get_dummies(y)
# 3개의 품종을 하나의 컬럼으로 만들고 일치한다면 1(T), 그렇지 않다면 0(F)이 나옴
print(y[0:5])

# 소프트맥스

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 모델 설정

model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))
'''
출력되는 개수가 3개, activation이 softmax가 된 것이 전에 만든 모델과의 차이점
softmax : 3개의 출력값의 합이 1이 됨, 그 3개의 값 중 가장 높은것이 예측된다고 생각하면 될 듯
'''


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x, y, epochs=30, batch_size=5)