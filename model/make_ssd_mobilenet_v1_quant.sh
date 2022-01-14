#! /bin/bash

./export_ssd_mobilenet_v1_quant.sh

tflite_convert --graph_def_file ssd_mobilenet_v1_quant_exported_graph/tflite_graph.pb --output_file tflite_model/ssd_mobilenet_v1_quant.tflite --input_shape "1,300,300,3" --input_arrays normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --inference_type=QUANTIZED_UINT8 --mean_values=128 --std_dev_values=128 --change_concat_input_ranges=false --allow_custom_ops
