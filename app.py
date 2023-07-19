from flask import Flask, request

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'No file uploaded', 400

    file = request.files['image']

    if file.filename == '':
        return 'No selected file', 400


    return 'File uploaded successfully'

if __name__ == '__main__':
    app.run()
