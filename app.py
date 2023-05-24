import torch
from seibetsu import transform, Net
from flask import Flask, request, render_template, redirect
import io
from PIL import Image
import base64
import cv2
from flask import Response
from torchvision import transforms
import pytesseract
from io import BytesIO
import pyocr
import re
import datetime
import easyocr
import numpy as np
from mynum import detect_gender
from mynum import detect_birthdate
import os
from flask import url_for

__author__ = 'Rick Torzynski <ricktorzynski@gmail.com>'
__source__ = ''
app = Flask(__name__)
UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# 最初の画面
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        image_data_url = request.form['imageData'] #cam
        image_data = re.sub('^data:image/.+;base64,', '', image_data_url) #cam
        image = Image.open(BytesIO(base64.b64decode(image_data))) #cam

        pred = predict(image)
        seibetsuDanjyo_ = getDanjyo(pred)
        del pred
        return render_template('result.html', seibetsuDanjyo=seibetsuDanjyo_, image=image_data_url) #cam

    elif request.method == 'GET':
        return render_template('webcam.html')

# 男女の予測メソッド
def predict(img):
    net = Net().cpu().eval()
    net.load_state_dict(torch.load('seibetsujudge_cpu_4.pt', map_location=torch.device('cpu')))
    transform = transforms.Compose([
        transforms.Resize(size=(100, 100)),
        transforms.ToTensor(),
    ])
    img = transform(img)
    img = img.unsqueeze(0)
    y = torch.argmax(net(img), dim=1).cpu().detach().numpy()
    return y

# 男女のラベル付けメソッド
def getDanjyo(label):
    if label == 0:
        return '女性'
    elif label == 1:
        return '男性'

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen.get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mynum', methods=['GET', 'POST'])
def mynum():
    if request.method == 'POST':
        # リクエストから画像ファイルを取得
        img_file = request.files['filename2']

        # OCRツールの初期化
        reader = easyocr.Reader(['ja'], gpu=False, model_storage_directory='ocr_model')
        tools = pyocr.get_available_tools()
        tool = tools[0]

        # OpenCVを使用して画像を読み込む
        img_bytes = img_file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img_gen = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        # EasyOCRを使用して性別を抽出
        img_gray = cv2.cvtColor(img_gen, cv2.COLOR_BGR2GRAY)
        txt_gender = reader.readtext(img_gray)
        gender = detect_gender(txt_gender)

        #PyOCRを使用して生年月日を抽出
        img_pil = Image.open(io.BytesIO(img_bytes))
        img_pil = img_pil.convert('L')
        txt_age = tool.image_to_string(img_pil, lang='jpn+eng', builder=pyocr.builders.TextBuilder(tesseract_layout=6))

        age = detect_birthdate(txt_age)

        # 結果をレンダリングするテンプレートに渡してレスポンスを返す
        # return render_template('result_mynum.html', age=age)
        return render_template('result_mynum.html', gender=gender, age=age)
    else:
        # テンプレートの読み込み
        return render_template('mynum.html')

if __name__ == '__main__':
  port = int(os.environ.get('PORT', 5000))
  app.run(debug=True, host='0.0.0.0', port=port)
  # app.run(host="0.0.0.0", port=5000, debug=True)
