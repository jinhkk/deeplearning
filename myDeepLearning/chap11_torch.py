import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 1. 데이터 로딩 및 분리 (Keras: df.iloc[:, 0:8], df.iloc[:, 8])
df = pd.read_csv('./data/pima-indians-diabetes3.csv')

x_data = df.iloc[:, 0:8].values.astype(np.float32)  # 입력 8개
y_data = df.iloc[:, 8].values.astype(np.float32).reshape(-1, 1)  # 출력 1개 (binary)

'''
| 코드                    | 설명                          |
| --------------------- | --------------------------- |
| `df.iloc[:, 0:8]`     | DataFrame에서 입력용 열(0\~7번) 선택 |
| `.values`             | Pandas → Numpy 배열로 변환       |
| `.astype(np.float32)` | float64 → float32 형으로 변환    |

'''


'''
import numpy as np

a = np.array([1, 2, 3, 4])  # shape: (4,) → 1차원
b = a.reshape(-1, 1)        # shape: (4, 1) → 2차원

print(b)
# [[1]
#  [2]
#  [3]
#  [4]]


-1은 자동으로 크기 계산해줘! 라는 의미

1은 열(column)의 개수를 지정한 것

즉,

"행 개수는 모르겠고 열 1개짜리 2차원 배열로 바꿔줘"

라는 뜻
'''
# 2. 텐서로 변환
x_tensor = torch.tensor(x_data)
y_tensor = torch.tensor(y_data)

# 3. TensorDataset 및 DataLoader 정의 (Keras의 batch_size=5와 유사하게)
dataset = TensorDataset(x_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=5, shuffle=True)


# 4. 모델 정의 (Keras의 Dense 12 -> 8 -> 1 계층과 대응)
class PimaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 12)  # input_dim=8 → hidden=12
        self.layer2 = nn.Linear(12, 8)  # hidden=12 → hidden=8
        self.output = nn.Linear(8, 1)  # hidden=8 → output=1

    def forward(self, x):
        x = torch.relu(self.layer1(x))  # Dense(12, activation='relu')
        x = torch.relu(self.layer2(x))  # Dense(8, activation='relu')
        x = torch.sigmoid(self.output(x))  # Dense(1, activation='sigmoid')
        return x


model = PimaClassifier()

# 5. 손실 함수 및 옵티마이저 정의 (Keras의 compile과 대응)
criterion = nn.BCELoss()  # binary_crossentropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. 모델 학습 (Keras: model.fit(x, y, epochs=100, batch_size=5))
for epoch in range(100):
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 정확도 계산 (output > 0.5 이면 1로 판단)
        predicted = (outputs > 0.5).float()
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    acc = correct / total * 100
    print(f"Epoch {epoch + 1}/100 - Loss: {total_loss:.4f} - Accuracy: {acc:.2f}%")
