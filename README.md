# Computer Vision
A collection of computer vision models and projects, specifically covering image classification and object detection.

![alt text](https://fraser.love/content/images/size/w2000/2023/07/intro-into-object-detection.jpg)
## Installation and Setup
Setup your environment and install the required dependencies as follows:

1. **Clone the Repository:**
```bash
git clone https://github.com/fraserlove/computer-vision.git
cd computer-vision
```

2. **Create a Python Virtual Environment:**
```bash
python -m venv .venv
source .venv/bin/activate
```

3. **Install Dependencies via PIP:**

```bash
pip install -r requirements.txt
```

4. **Run a Jupyter Notebook server**
``` bash
jupyter notebook
```
## Included Models

- Image Classification
    - Binary Classifier
    - Multi-label Classifier (Feed-Forward Neural Network)
    - Multi-label Classifier (CNN)
    - Multi-label Classifier (Transfer Learning with Pre-Trained EfficientNet Model from Keras Applications)
- Object Detection
    - Inference with the TensorFlow Object Detection API and TensorFlow Hub
    - Fine Tuning with the TensorFlow Object Detection API
    - Yolo NAS
    - Yolo v8

## A Note on Datasets
Kaggle is used for downloading datasets. Set up an account and generate an API key. Then enter the following,
replacing `USERNAME` and `API_KEY` with their values.
```
mkdir ~/.kaggle
echo 'api_token = {"username":USERNAME,"key":API_KEY}' >> ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```
