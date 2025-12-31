# 参考教程https://doc.embedfire.com/linux/rk356x/Python/zh/latest/ai/resnet18_pytorch.html
# Import the ONNX mapping patch first to handle compatibility issues
import onnx_mapping_patch

import numpy as np
import cv2
from rknn.api import RKNN
import torchvision.models as models
import torch
import os, onnx
import argparse

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def torch_version():
    import torch
    torch_ver = torch.__version__.split('.')
    torch_ver[2] = torch_ver[2].split('+')[0]
    return [int(v) for v in torch_ver]

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, 
            default='./saved_model/plate_rec_color.onnx', help='onnx path')
    parser.add_argument('--save_path', type=str, 
            default='./saved_model/plate_rec_color.rknn', help='rknn save path')
    # parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    print(opt)

    if torch_version() < [1, 9, 0]:
        import torch
        print("Your torch version is '{}', in order to better support the Quantization Aware Training (QAT) model,\n"
              "Please update the torch version to '1.9.0' or higher!".format(torch.__version__))
        exit(0)


    batch_size = 1
    model = onnx.load(opt.onnx)
    inp = model.graph.input[0]
    print("Input name:", inp.name)
    for i, dim in enumerate(inp.type.tensor_type.shape.dim):
        if i == 0:
            batch_size = dim.dim_value if dim.dim_value != 0 else dim.dim_param
            print(f"  Batch size: {batch_size}")
        else:
            print(f"  Dim {i}: {dim.dim_value if dim.dim_value != 0 else dim.dim_param}")

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config
    print('--> Config model')
    #rknn.config(target_platform='rk3588')
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform='rk3588',
        optimization_level=3,        # 最高优化级别
    )
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(
        model = opt.onnx,
        inputs = ['images'],  # Specify the input name
        input_size_list = [[batch_size, 3, 48, 168]]  # [batch_size, channels, height, width]
    )
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(opt.save_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')


    # Set inputs
    img = cv2.imread('./0_125.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to the expected input size: 168x48 (width x height) as per the model
    img = cv2.resize(img, (168, 48))  # (width, height)
    #img = np.expand_dims(img, 0)
    
    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    """
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    np.save('./pytorch_resnet18_qat_0.npy', outputs[0])
    #show_outputs(softmax(np.array(outputs[0][0])))
    print(outputs)
    print('done')
    """

    rknn.release()
