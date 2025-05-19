import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 1. 데이터 불러오기
data_set = np.loadtxt("./data/ThoraricSurgery3.csv", delimiter=",")
x = data_set[:, 0:16]
y = data_set[:, 16]

# 2. 넘파이 배열을 텐서로 변환 (float32, long 필요)
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # (N,) → (N, 1)

# 3. Dataset과 DataLoader 생성
dataset = TensorDataset(x_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 4. 모델 정의 (Sequential 방식 사용)
model = nn.Sequential(
    nn.Linear(16, 30),    # 입력 16 → 은닉층 30
    nn.ReLU(),
    nn.Linear(30, 1),     # 출력 1 (이진분류)
    nn.Sigmoid()
)

# 5. 손실 함수 및 옵티마이저 정의
criterion = nn.BCELoss()  # binary_crossentropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. 학습 루프
epochs = 5
for epoch in range(epochs):
    total_loss = 0.0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
