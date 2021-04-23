# 环境配置

    ubuntu：18.04

    cuda：11.0

    cudnn：8.0

    tensorrt：7.2.16

    OpenCV：3.4.2

    cuda，cudnn，tensorrt和OpenCV安装包（编译好了，也可以自己从官网下载编译）可以从链接: https://pan.baidu.com/s/1dpMRyzLivnBAca2c_DIgGw 密码: 0rct

    cuda安装

    如果系统有安装驱动，运行如下命令卸载

    sudo apt-get purge nvidia*

    禁用nouveau，运行如下命令

    sudo vim /etc/modprobe.d/blacklist.conf

    在末尾添加

    blacklist nouveau

    然后执行

    sudo update-initramfs -u

    chmod +x cuda_11.0.2_450.51.05_linux.run

    sudo ./cuda_11.0.2_450.51.05_linux.run

    是否接受协议: accept

    然后选择Install

    最后回车

    vim ~/.bashrc 添加如下内容：

    export PATH=/usr/local/cuda-11.0/bin:$PATH

    export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH

    source .bashrc 激活环境

    cudnn 安装

    tar -xzvf cudnn-11.0-linux-x64-v8.0.4.30.tgz

    cd cuda/include

    sudo cp *.h /usr/local/cuda-11.0/include

    cd cuda/lib64

    sudo cp libcudnn* /usr/local/cuda-11.0/lib64

    tensorrt及OpenCV安装

    定位到用户根目录

    tar -xzvf TensorRT-7.2.1.6.Ubuntu-18.04.x86_64-gnu.cuda-11.0.cudnn8.0.tar.gz

    cd TensorRT-7.2.1.6/python，该目录有4个python版本的tensorrt安装包

    sudo pip3 install tensorrt-7.2.1.6-cp37-none-linux_x86_64.whl（根据自己的python版本安装）

    pip install pycuda 安装python版本的cuda

    定位到用户根目录

    tar -xzvf opencv-3.4.2.zip 以备推理调用
    
# yolov5s模型转换onnx

    pip install onnx

    pip install onnx-simplifier

    git clone https://github.com/ultralytics/yolov5.git

    cd yolov5/models

    vim common.py

    把BottleneckCSP类下的激活函数替换为relu，tensorrt对leakyRelu int8量化不稳定（这是一个深坑，大家记得避开）即修改为self.act = nn.ReLU(inplace=True)

    训练得到模型后

    cd yolov5

    python models/export.py --weights 训练得到的模型权重路径 --img-size 训练图片输入尺寸

    python3 -m onnxsim onnx模型名称 yolov5s-simple.onnx 得到最终简化后的onnx模型
    
# onnx模型转换为 int8 tensorrt引擎

    git clone https://github.com/Wulingtian/yolov5_tensorrt_int8_tools.git（求star）

    cd yolov5_tensorrt_int8_tools

    vim convert_trt_quant.py 修改如下参数

    BATCH_SIZE 模型量化一次输入多少张图片

    BATCH 模型量化次数

    height width 输入图片宽和高

    CALIB_IMG_DIR 训练图片路径，用于量化

    onnx_model_path onnx模型路径

    python convert_trt_quant.py 量化后的模型存到models_save目录下
    
# tensorrt模型推理

    git clone https://github.com/Wulingtian/yolov5_tensorrt_int8.git（求star）

    cd yolov5_tensorrt_int8

    vim CMakeLists.txt

    修改USER_DIR参数为自己的用户根目录

    vim http://yolov5s_infer.cc 修改如下参数

    output_name1 output_name2 output_name3 yolov5模型有3个输出

    我们可以通过netron查看模型输出名

    pip install netron 安装netron

    vim netron_yolov5s.py 把如下内容粘贴

        import netron

        netron.start('此处填充简化后的onnx模型路径', port=3344)

    python netron_yolov5s.py 即可查看 模型输出名

    trt_model_path 量化的的tensorrt推理引擎（models_save目录下trt后缀的文件）

    test_img 测试图片路径

    INPUT_W INPUT_H 输入图片宽高

    NUM_CLASS 训练的模型有多少类

    NMS_THRESH nms阈值

    CONF_THRESH 置信度

    参数配置完毕

    mkdir build

    cd build

    cmake ..

    make

    ./YoloV5sEngine 输出平均推理时间，以及保存预测图片到当前目录下，至此，部署完成！
