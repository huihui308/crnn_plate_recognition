# 参考教程https://doc.embedfire.com/linux/rk356x/Python/zh/latest/ai/resnet18_pytorch.html
# Import the ONNX mapping patch first to handle compatibility issues

import sys
import os

# 当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 上一级目录
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# 放到最前，避免同名模块冲突
sys.path.insert(0, parent_dir)

from alphabets import plate_chr
from demo import decodePlate
from onnx_infer import plate_color_list

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
    img_path1 = './test_images/0_127.jpg'
    img_path2 = './test_images/0_125.jpg'  # Use the original image as second image

    # Load first image
    if not os.path.exists(img_path1):
        raise ValueError(f"Warning: Image {img_path1} not found.")
    img1 = cv2.imread(img_path1)
    if img1 is None:
        raise ValueError(f"Warning: Image {img_path1} not found.")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    # Load second image
    if not os.path.exists(img_path2):
        raise ValueError(f"Warning: Image {img_path1} not found.")
    img2 = cv2.imread(img_path2)
    if img2 is None:
        raise ValueError(f"Warning: Image {img_path1} not found.")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Resize to the expected input size: 168x48 (width x height) as per the model
    img1 = cv2.resize(img1, (168, 48))  # (width, height)
    img2 = cv2.resize(img2, (168, 48))  # (width, height)
    
    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Prepare batch input for inference
    # The model expects a batch of images based on the batch size read from ONNX
    if isinstance(batch_size, int) and batch_size == 1:
        # If batch size is 1, just use one image
        print('--> Running model for image 1')
        preds = rknn.inference(inputs=[img1])

        # Process results for the image
        color_results = preds[1]  # Get color result
        plate_result = preds[0]   # Get plate result

        # Convert color prediction to specific color name
        color_index = np.argmax(color_results, axis=1)[0]  # Get the index of the highest probability
        plate_color = plate_color_list[color_index]
        print("Plate color:", plate_color)

        # Use numpy argmax instead of torch argmax
        pred = np.argmax(plate_result, axis=2)  # Use axis=2 instead of dim=2
        # print(preds)
        # Reshape and convert to flat array
        pred = pred.flatten()  # Use numpy flatten instead of torch view(-1).detach().cpu().numpy()
        newPreds = decodePlate(pred)
        plate = ""
        for i in newPreds:
            plate += plate_chr[int(i)]
        print(f'Recognized plate: {plate} -- {plate_color}')
    else:
        # If batch size is greater than 1, we need to provide a batch of images
        # For simplicity, we'll duplicate the same image to fill the batch
        batch_images = []
        for i in range(batch_size if isinstance(batch_size, int) else 1):
            if i % 2 == 0:
                batch_images.append(img1)
            else:
                batch_images.append(img2)

        print(f'--> Running model with batch size {len(batch_images)}')
        print(f'Input image shape: {batch_images[0].shape}')

        # Stack the images to create the correct batch format (batch_size, height, width, channels)
        # The model expects input in NHWC format: (batch_size, height, width, channels)
        stacked_images = np.stack(batch_images, axis=0)
        print(f'Stacked images shape: {stacked_images.shape}')

        # RKNN inference expects images in the format that matches the model's input
        # Since the model was built with batch size 2, we need to provide 2 images in the correct format
        preds = rknn.inference(inputs=[stacked_images])

        # Process results for each image in the batch
        color_results = preds[1]  # Get color results [batch_size, num_colors]
        plate_result = preds[0]   # Get plate results [batch_size, sequence_length, num_classes]

        for i in range(len(batch_images)):
            # Process color for this image in the batch
            color_index = np.argmax(color_results[i:i+1], axis=1)[0]  # Get the index of the highest probability
            plate_color = plate_color_list[color_index]
            print(f"Plate color (image {i+1}):", plate_color)

            # Use numpy argmax for this image in the batch
            pred = np.argmax(plate_result[i:i+1], axis=2)  # Use axis=2 instead of dim=2
            # print(preds)
            # Reshape and convert to flat array
            pred = pred.flatten()  # Use numpy flatten instead of torch view(-1).detach().cpu().numpy()
            newPreds = decodePlate(pred)
            plate = ""
            for j in newPreds:
                plate += plate_chr[int(j)]
            print(f'Recognized plate (image {i+1}): {plate} -- {plate_color}')

    print('done')

    rknn.release()
