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

## Project Architecture

## Network Architecture 
 
## Datasets 

## Performance

## TODO
 - [ ] iou threshold을 70 이상으로 변경하기   

## Copyright 