from flask import Flask, render_template, request
from fastai.transforms import *
from fastai.dataset import *
import re
import base64

PATH = "/home/ubuntu/Digit/Model/"

def load_model():
	print('Running')
	global model, val_tfm
	sz=28
	arch=resnet34
	model = torch.load(f'{PATH}mnist')
	model = model.cpu().eval()
	_, val_tfm = tfms_from_model(arch, sz)
	print('Loaded')
	
load_model()

app = Flask(__name__)
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict/', methods=['GET','POST'])
def predict():
	parseImage(request.get_data())
	image = VV(T(val_tfm(open_image('output.png'))[None])).cpu()
	preds = np.argmax(model(image).data.numpy())
	print(preds)
	return str(preds)
    
    
def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))

if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
