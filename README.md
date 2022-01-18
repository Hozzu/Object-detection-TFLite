# Object-detection-TFLite
Object detection application using TensorFlow Lite

## What is this?
This application is made for testing the speed and accuracy of object detection deep learning model.

This appliation supports two mode: camera and image.  
Camera mode runs object detection from camera input and displays the result on monitor. This shows the speed and accuracy of model intuitively and visually.  
Image mode runs object detection with given image files and prints the average inference speed and mAP (mean average precision). This shows the speed and accuracy of model quantitatively.  

## Build
This application runs on any HW supporting TensorFlow Lite with GPU and hexagon delegates.  

TensorFlow Lite with delegates are required. If your system is aarch64 Linux, check https://github.com/Hozzu/TFLite-aarch64-linux-with-delegate.  
opencv (only core and imgproc) and qcarcam client libraries for aarch64 Linux are included. If your system is not aarch64 Linux, you will need these libraries.  
Qcarcam (ais_client), fastcv, json-c, jpeg libraries are additionally required.  
Python3.x and python modules (collections, numpy, math, json) are requried for calculating mAP.  

After checking theses dependencis, build app using make.  
It builds pkshin_detect binary.  

## How to use

Type pkshin_detect --help to show guide.  

run_camea.sh is an example running camera mode.  
run_mAP.sh is an example running image mode and calculating mAP.  

SSD Mobilenet V1 and V2, COCO2017 validation data set and annotations are included.  
Go to model directory to see how I converted the model. 

You can of course change the model file and data set by yourself.

## Result

It was tested on SA8195 running Automotive Grade Linux.  

![image](https://user-images.githubusercontent.com/28533445/149861427-d533085a-8688-469e-b5bb-6bb9819897bd.png)

