export LD_LIBRARY_PATH=./lib

echo "Detecting images.."
./pkshin_detect image model/ssd_mobilenet_v2_quant.tflite coco_labels.txt coco_val2017 detection_result.json npu

if [ $? -eq 1 ];
then
	echo "Calculating mAP.."
	python3 python_src/main.py coco_val2017.json detection_result.json
fi
