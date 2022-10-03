import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable
import lib.utils.utils as utils
import lib.models.crnn as crnn
import lib.config.alphabets as alphabets
import yaml
from easydict import EasyDict as edict
import argparse
 
def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='lib/config/360CC_config.yaml')
    parser.add_argument('--image_path', type=str, default='/mnt/Gpan/Mydata/pytorchPorject/myCrnnPlate/新AU3006_convert0177.jpg', help='the path to your image')
    parser.add_argument('--checkpoint', type=str, default='/mnt/Gpan/Mydata/pytorchPorject/myCrnnPlate/output/360CC/crnn/2022-01-25-22-39/checkpoints/checkpoint_7_acc_0.8618.pth',
                        help='the path to your checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.plateName
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

def recognition(config, img, model, converter, device):

    # github issues: https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec/issues/211
    # h, w = img.shape
    # fisrt step: resize the height and width of image to (32, x)
    # img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.H / h, fy=48config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)

    # # second step: keep the ratio of image's text same with training
    # h, w = img.shape
    # w_cur = int(img.shape[1] / (config.MODEL.IMAGE_SIZE.OW / config.MODEL.IMAGE_SIZE.W))
    # img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=1.0, interpolation=cv2.INTER_CUBIC)
    # img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, w_cur, 1))

    img = cv2.resize(img, (168,48))
    img = np.reshape(img, (48, 168, 3))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    preds = model(img)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    print('results: {0}'.format(sim_pred))

if __name__ == '__main__':

    config, args = parse_arg()
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device =torch.device('cpu')
    model = crnn.get_crnn(config).to(device)
    print('loading pretrained model from {0}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint,map_location=device)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    started = time.time()

    img_raw = cv2.imread(args.image_path)
    img =img_raw
    # img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)

    recognition(config, img, model, converter, device)

    # cv2.imshow('raw', img_raw)
    # cv2.waitKey(0)

    finished = time.time()
    print('elapsed time: {0}'.format(finished - started))

