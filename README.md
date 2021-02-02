# yolov5_tensorrt_int8_tools
tensorrt int8 量化yolov5 4.0 onnx模型
onnx模型转换为 int8 tensorrt引擎

git clone https://github.com/Wulingtian/yolov5_tensorrt_int8_tools.git（求star）

cd yolov5_tensorrt_int8_tools

vim convert_trt_quant.py 修改如下参数

BATCH_SIZE 模型量化一次输入多少张图片

BATCH 模型量化次数

height width 输入图片宽和高

CALIB_IMG_DIR 训练图片路径，用于量化

onnx_model_path onnx模型路径

python convert_trt_quant.py 量化后的模型存到models_save目录下
