#!/bin/bash
PIPELINE_CONFIG_PATH=~/Downloads/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18/pipeline.config
TRAINED_CKPT_PREFIX=~/Downloads/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18/model.ckpt
EXPORT_DIR=~/ssd_mobilenet_v1_quant_exported_graph
pushd ~/models/research
python object_detection/export_tflite_ssd_graph.py \
--pipeline_config_path=${PIPELINE_CONFIG_PATH} \
--trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
--output_directory=${EXPORT_DIR} \
--max_detections=100
popd
