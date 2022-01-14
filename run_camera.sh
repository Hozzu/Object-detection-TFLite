export LD_LIBRARY_PATH=./lib

./pkshin_detect camera model/ssd_mobilenet_v2_quant.tflite coco_labels.txt ssdDisplay.xml npu
