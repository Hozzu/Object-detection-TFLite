Requirements:  
ubuntu 18.04  
python 3.6  
protobuf-compiler 3.0  
tensorflow 1.15.5  
  
  
Guide:  
git clone https://github.com/tensorflow/models.git -b r2  
cd models/research  
protoc object_detection/protos/*.proto --python_out=.  
cp object_detection/packages/tf1/setup.py .  
python -m pip install --use-feature=2020-resolver .  
python object_detection/builders/model_builder_tf1_test.py  

copy all the scripts in home directory  
run the make scripts in home directory  
