from flask import Flask, request, render_template, redirect
from flask import session
import os
import pickle
import torch

app = Flask(__name__)

# Model을 Load하는 함수 정의

def load_model():  # 연근 및 등급 분류 모델
    global insam_model
    model_path = "C:\Model\Age_Grade_best.pt"

    with open(model_path, "rb") as f:
        insam_model = pickle.load(f)

# 플라스크 애플리케이션 초기화 함수 정의
@app.before_first_request
def before_first_request():
    load_model()

@app.route('/')
def main():
    return render_template("insam.html")

@app.route('/upload', methods=['POST'])
def upload():
    image = request.files['image']
    result = apply_model(image)

    return render_template('result.html', result=result)

def apply_model(image):

    return image.filename

# app 실행하는 main()
if __name__ == '__main__':
    app.run()