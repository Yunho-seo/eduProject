from flask import Flask, request, render_template
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import requests

app = Flask(__name__)

# 모델 클래스 정의 (PyTorch 모델을 사용하는 경우)
class YourModel(nn.Module):
    def __init__(self, num_classes=80):  # num_classes는 분류할 클래스의 개수입니다.
        super(YourModel, self).__init__()

        # YOLOv5와 유사한 합성곱 레이어 정의
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

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

def load_yolov5_model():
    global yolov5_model
    model_path = "C:/model/YOLOv5.pt"  # YOLOv5 사전 학습 가중치 파일 경로
    yolov5_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)

def load_model():
    global insam_model
    model_path = "C:/model/Age_Grade_best.pt"

    # 모델 클래스 인스턴스화
    insam_model = YourModel()

    # 저장된 state_dict 로드
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    insam_model.load_state_dict(checkpoint)

    # 모델을 평가 모드로 설정
    insam_model.eval()

# 플라스크 애플리케이션 초기화 함수 정의
@app.before_first_request
def before_first_request():
    load_yolov5_model()
    load_model()

@app.route('/')
def main():
    return render_template("insam.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    # 이미지 파일을 받아와 임시 폴더에 저장
    image = request.files['image']
    temp_image_path = "temp_image.jpg"
    image.save(temp_image_path)

    # YOLOv5 모델 적용
    yolov5_result = apply_yolov5(temp_image_path)

    # 다른 모델 적용 (이전 코드에서의 apply_model 함수 호출)
    other_model_result = apply_model(temp_image_path)

    # 임시 이미지 파일 삭제
    os.remove(temp_image_path)

    # 결과를 딕셔너리로 반환
    result = {
        "yolov5_result": yolov5_result,
        "other_model_result": other_model_result
    }

    return render_template('result.html', result=result)

def apply_yolov5(image_path):
    # YOLOv5를 사용하여 이미지에서 객체 검출 수행
    img = Image.open(image_path)

    # 이미지를 YOLOv5 모델에 적용하여 결과를 얻습니다.
    results = yolov5_model(img)

    # 결과 반환 (여기서는 문자열로 표시)
    return "YOLOv5 Model applied successfully"

def apply_model(image_path):
    # 이미지를 모델에 적용하여 결과를 얻는 로직을 구현
    # 이 부분에서는 insam_model을 사용하여 이미지를 모델에 전달하고 결과를 얻습니다.
    # PyTorch를 사용하여 이미지를 전처리하고 모델에 적용하는 과정이 필요합니다.

    # 예시: PyTorch를 사용하여 이미지를 전처리하는 방법
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # 배치 차원 추가
    with torch.no_grad():
        output = insam_model(image)

    return "Model applied successfully"

# app 실행하는 main()
if __name__ == '__main__':
    app.run(debug=True)