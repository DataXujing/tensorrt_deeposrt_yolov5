## TensorRT DeepSort的C++实现



### 1.环境

+ TensorRT 7.2.3
+ win10
+ VS2017

### 2.ReID模型的TensorRT实现

#1.ReID项目下载

```shell
git clone git@github.com:ZQPei/deep_sort_pytorch.git
git clone git@github.com/JDAI-CV/fast-reid.git
```

#2.合并项目生成ONNX

```shell
#1.
将model/exportOnnx.py拷贝到deepsort_sort_pytorch项目下
#2.
将fast-reid下的fastreid文件夹拷贝到deep_sort_pytorch项目下
#3.
安装必要的python库，主要来源于上述两个项目中的requirements.txt

```

```shell
#4.
cd deepsort_sort_pytorch
python exportOnnx.py

# 生成Dynamic shape的 deepsort.onnx 文件
```

#3.ONNX生成TensorRT Engine

```shell
trtexec --onnx=deepsort.onnx --saveEngine=deepsort.trt --workspace=1024 --minShapes=input:1x3x128x64 --optShapes=input:128x3x128x64 --maxShapes=input:128x3x128x64 --fp16 --verbose
```



### 3.解耦合的目标检测

#1.解耦的目标检测模型

项目是使用了YOLOV5s 3.0的模型。该部分TensorRT加速和DeepSort在项目设计上是是解耦的，因此读者可以替换为任何感兴趣的目标检测模型

#2.YOLOV5s 3.0核心代码的修改

这部分代码在model/yolo.py和model/export_xxxx.py中

#3.生成ONNX

```shell
cd yolov5
python export_xxx.py

# 生成ONNX
```

#4.ONNX转TensorRT Engine

```shell
trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s.engine --verbose
```

### 4.Slide教程

+ 打开[Slide教程](./tensorrt_deepsort.pdf)详细解读
+ VS2017直接build