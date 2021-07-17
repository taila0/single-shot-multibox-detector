# Simple Single Shot MultiBox Detector 

이 레파지토리에서는 Single Shot Multibox Detector의 아키텍쳐를 사용해  
이미지 내 여러 객체의 위치 정보와 class 정보를 추출하는 Detection 알고리즘을 구현합니다.      
  
[Single Shot Multibox Detector](https://arxiv.org/abs/1512.02325) `one stage` 아키텍쳐 입니다.
박스는 RCNN 계열의 two stage 아키텍쳐와 달리 `ROI Pooling` 을 통한 후보군 추출 단계가 없고
바로 객체의 위치와 분류합니다. 
이를 통해 two-stage 계열의 detection 보다 속도가 빠른 장점이 있습니다.   

해당 레포지토리는 논문에서 사용한 데이터와 달리 detection 을 위한 mnist dataset 을 생성해 사용합니다.  
그렇기 때문에 `default boxes`(or `anchor`) 및 상세 요소들은 논문과 다름니다.            
해당 알고리즘은 TF2로 만들어져 있습니다.

![그림](https://i.imgur.com/GwqRK5A.jpg)

## Table of Contents
 - Requirements
 - Project Architecture
 - Network Architecture
 - Datasets
 - Training
 - Detail implements  
 - Performance
 

## Requirements 

<details>
  <summary> Requirements </summary>
  absl-py==0.11.0
appnope==0.1.2
astunparse==1.6.3
backcall==0.2.0
cached-property==1.5.2
cachetools==4.2.1
certifi==2020.12.5
chardet==4.0.0
cycler==0.10.0
decorator==5.0.9
et-xmlfile==1.1.0
flatbuffers==1.12
gast==0.4.0
google-auth==1.27.0
google-auth-oauthlib==0.4.2
google-pasta==0.2.0
grpcio==1.34.1
h5py==3.1.0
idna==2.10
imbalanced-learn==0.8.0
imblearn==0.0
importlib-metadata==3.7.0
ipykernel==5.5.5
ipython==7.23.1
ipython-genutils==0.2.0
jedi==0.18.0
joblib==1.0.1
jupyter-client==6.1.12
jupyter-core==4.7.1
Keras==2.4.3
keras-nightly==2.5.0.dev2021032900
Keras-Preprocessing==1.1.2
kiwisolver==1.3.1
Markdown==3.3.4
matplotlib==3.4.2
matplotlib-inline==0.1.2
mpmath==1.2.1
numpy==1.19.5
oauthlib==3.1.0
opencv-python==4.5.1.48
openpyxl==3.0.7
opt-einsum==3.3.0
pandas==1.2.3
parso==0.8.2
pexpect==4.8.0
pickle5==0.0.11
pickleshare==0.7.5
Pillow==8.2.0
prompt-toolkit==3.0.18
protobuf==3.15.3
ptyprocess==0.7.0
pyasn1==0.4.8
pyasn1-modules==0.2.8
Pygments==2.9.0
pyparsing==2.4.7
python-dateutil==2.8.1
pytz==2021.1
PyYAML==5.4.1
pyzmq==22.0.3
requests==2.25.1
requests-oauthlib==1.3.0
rsa==4.7.2
scikit-learn==0.24.2
scipy==1.6.3
seaborn==0.11.1
six==1.15.0
sklearn==0.0
sympy==1.8
tensorboard==2.5.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.0
tensorflow==2.5.0
tensorflow-estimator==2.5.0
termcolor==1.1.0
threadpoolctl==2.1.0
tornado==6.1
tqdm==4.58.0
traitlets==5.0.5
typing-extensions==3.7.4.3
urllib3==1.26.3
wcwidth==0.2.5
Werkzeug==1.0.1
wget==3.2
wrapt==1.12.1
xgboost==1.4.2
xlrd==2.0.1
zipp==3.4.0

  
</details>


## Project Architecture

## Network Architecture 
 
## Datasets 

## Performance

## TODO
 - [ ] iou threshold을 70 이상으로 변경하기   

## Copyright 
