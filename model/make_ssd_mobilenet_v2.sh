#! /bin/bash

./export_ssd_mobilenet_v2.sh

tflite_convert --graph_def_file ssd_mobilenet_v2_exported_graph/tflite_graph.pb --output_file tflite_model/ssd_mobilenet_v2.tflite --input_shape "1,300,300,3" --input_arrays normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --inference_type=FLOAT --allow_custom_ops
