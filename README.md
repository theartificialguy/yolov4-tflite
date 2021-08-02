# yolov4-tflite
Run yolov4-tf-tflite on pre-trained or custom weights and classes.

## Steps to run:
  1) You will need custom weights and classes file in order to run yolov4-tf or yolov4-tflite on your own custom classes. (You can run on pre-trained weights and classes too)
  2) Download the custom weights and classes files.
  3) Clone the repository: git clone https://github.com/yash-007/yolov4-tflite.git
  4) Copy paste custom weights file in "yolov4-tflite/data" folder and obj.names in "yolov4-tflite/data/classes" folder.
  5) Navigate to "yolov4-tflite/core/" and open config.py and change the line where the classes file is mentioned to your custom class names file i.e. obj.names.
  6) Convert the yolov4 custom weights file to tensorflow weight file using the below command.
  7) python save_model.py --weights ./data/custom.weights --output ./checkpoints/custom-416 --input_size 416 --model yolov4
  8) Convert the custom yolov4-tf to tflite weights using the below command.
  9) python convert_tflite.py --weights ./checkpoints/custom-416 --output ./checkpoints/custom-416.tflite
  10) Run the detector using: python detect.py
