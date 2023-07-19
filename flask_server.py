from flask import Flask, request, render_template, redirect
# request : 클라이언트 요청을 파싱(parsing 분해)
# rendering : (그래픽) html 파일
# redirect : 다른 페이지로 전송할 때 사용
import os
from flask import session  # 세션(session) : 네트워크에서 접속하는 사람마다 만들어지는 객체 (세션 객체)
app = Flask(__name__)
app.secret_key = 'You Will Never Guess'  # 요청
@app.route('/')
def index():
    session['key'] = 0  # 세션 변수를 생성
    return render_template("index.html")  # html 페이지를 jinja2 문법으로 해석하여 template으로 보낸다.

@app.route('/image_map')
def image_map():
    return render_template("image_map.html")

@app.route("/table1")
def table1():
    return render_template("table1.html")

@app.route("/table2")
def table2():
    return render_template("table2.html")

@app.route("/carousel")
def carousel():
    return render_template("carousel.html")

@app.route("/player")
def player():
    return render_template("player.html")

@app.route("/stadium")
def stadium():
    return render_template("stadium.html")


if __name__ == '__main__':
    app.run(debug=True)