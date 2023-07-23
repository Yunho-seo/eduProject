import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class YourModel(nn.Module):
    def __init__(self, num_classes=80):  # num_classes는 분류할 클래스의 개수입니다.
        super(YourModel, self).__init__()

        # YOLOv5와 유사한 합성곱 레이어 정의
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # 나머지 합성곱 레이어들도 유사하게 정의

        # YOLOv5와 유사한 Fully Connected 레이어 정의
        self.fc1 = nn.Linear(7 * 7 * 128, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # 합성곱 레이어 적용
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 나머지 합성곱 레이어 적용

        # Fully Connected 레이어 적용
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

model = YourModel()

# 저장된 state_dict 로드
model_path = "C:/model/Age_Grade_best.pt"
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

# state_dict의 키들 출력
print("State_dict의 키들:")
for key in model.state_dict():
    print(key)