# Computer Vision
A Collection of computer vision projects (image classification and object detection).

### Setup
```
git clone https://github.com/fraserlove/computer-vision.git
cd computer-vision
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Kaggle is used for downloading datasets. Set up an account and generate an API key. Then enter the following,
replacing `USERNAME` and `API_KEY` with their values.
```
mkdir ~/.kaggle
echo 'api_token = {"username":USERNAME,"key":API_KEY}' >> ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```
