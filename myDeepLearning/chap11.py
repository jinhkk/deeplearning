import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./data/pima-indians-diabetes3.csv')

print(df.head(5)) # 위에서 부터 5개
print(df.tail(5)) # 아래에서 부터 5개

print(df["diabetes"].value_counts())  # diabetes 속성의 값 : 0 과 1 을 세어서 나타내줌

# 각 정보별 특징을 좀 더 자세히 출력
print(df.describe())
print('=' * 50)
# 각 항목이 어느 정도의 상관 관계를 가지고 있는지 나타냄
print(df.corr())

#데이터 간의 상관 관계를 그래프로 표현
colormap = plt.cm.gist_heat # 그래프의 색상 구성을 정함 (gist_heat는 heat맵에서 자주 사용된다고 함.)
plt.figure(figsize=(12, 12)) # 그래프의 크기 설정

# 그래프의 속성 결정. vmax의 값을 0.5로 지정해 0.5에 가까울수록 밝은색으로 표시.
sns.heatmap(df.corr(), linewidths=0.1, vmax=0.5, cmap=colormap, linecolor='white', annot=True)
'''
df.corr() : 위의 df.corr()의 결과를 사용
linewidths : 선의 길이
vamx : 밝기 조절 ( 밝을수록 관련높음 )
cmap : 컬러맵( 아까 만든 컬러맵 사용 )
linecolor : 라인의 색 지정
annot : 숫자들이 보이게 True로 설정
'''
plt.show()
# plasma를 기준으로 각각 정상과 당뇨가 어느 정도 비율로 분포하는지 살펴봄
plt.hist(x=[df.plasma[df.diabetes==0], df.plasma[df.diabetes==1]], bins=30, histtype='barstacked', label=['normal', 'diabetes'])
'''
plt.hist : 히스토그램 생성
x=[df.plasma[df.diabetes==0], df.plasma[df.diabetes==1]] : x축에 어떤 요소를 불러올지 설정(위의 경우는 당뇨가 있냐/없냐 의 두개의 경우를 불러옴)
    여기서 df.plasma <- 이 plasma 위치가 기준으로 할 속성의 위치
bins : 하나의 그래프 안에 들어갈 막대의 개수
histtype : barstacked(두개의 그래프가 하나로 합쳐서 나오게 설정) 등 그래프의 타입을 정하는듯 함
label : 레이블 설정(막대의 이름 지정)
'''
plt.legend()
plt.show()

# 위의 그래프를 bmi 기준으로 다시 생성 후 확인
plt.hist(x=[df.bmi[df.diabetes==0], df.bmi [df.diabetes==1]], bins=30, histtype='barstacked', label=['normal', 'diabetes'])
plt.legend()
plt.show()

# 피마 인디언 당뇨병 예측 실행

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('./data/pima-indians-diabetes3.csv')

x = df.iloc[:,0:8]
y = df.iloc[:,8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu', name='Dense_1'))
model.add(Dense(8, activation='relu', name='Dense_2'))
model.add(Dense(1, activation='sigmoid', name='Dense_3'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x, y, epochs=100, batch_size=5)

# accuracy : 정확도
# 그러므로 76% 확률로 피마 인디언 당뇨병을 예측하는 모델을 만든거임