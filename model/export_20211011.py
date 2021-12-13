"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats
https://blog.csdn.net/linghu8812/article/details/109322729
Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse

import torch
import torch.nn as nn

import models
from models.experimental import attempt_load
from utils.activations import Hardswish,Swish,MemoryEfficientSwish,MemoryEfficientMish,FReLU
from utils.general import set_logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./YOLOV5x_20211011.pt', help='weights path')  # from yolov5/models/
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()

    # Input
    img = torch.zeros((opt.batch_size, 3, *opt.img_size))  # image size(1,3,320,192) iDetection

    # Load PyTorch model
    model = attempt_load(opt.weights, map_location=torch.device('cpu'))  # load FP32 model

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.Swish):
                m.act = Swish()
            elif isinstance(m.act, nn.MemoryEfficientSwish):
                m.act = MemoryEfficientSwish()  
            elif isinstance(m.act, nn.MemoryEfficientMish):
                m.act = MemoryEfficientMish() 
            elif isinstance(m.act, nn.FReLU):
                m.act = FReLU()        
        if isinstance(m, models.yolo.Detect):
            m.forward = m.forward  # assign forward (optional)
    model.model[-1].export = False  # set Detect() layer export=True  #《-----------------------
    # model.model[-1].export = False  # set Detect() layer export=True

    y = model(img)  # dry run

    # ONNX export
    try:
        import onnx
        from onnxsim import simplify

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        # torch.onnx.export(model, img, f, verbose=True, opset_version=10, input_names=['images'],
        #                   output_names=['output'] if y is None else ['output_0'])
        # torch.onnx.export(model, img, f, verbose=True, opset_version=10, input_names=['input_006'],
        #           output_names=['output_006_0',"output_006_1","output_006_2"])

        torch.onnx.export(model, img, f, verbose=True, opset_version=11, input_names=['input_yolov5'],
                  output_names=['output_yolov5'])

        # torch.onnx.export(model, img, f, verbose=False, opset_version=10, input_names=['images'],
        #                   output_names=['output',"1346","1366"])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, f)
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    print('\nExport complete. Visualize with https://github.com/lutzroeder/netron.')