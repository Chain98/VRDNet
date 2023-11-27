import os, time, argparse
import numpy as np
from PIL import Image
import torch
from torchvision.utils import save_image as imwrite
from utils import load_checkpoint, tensor2cuda
from model.models import VRDNet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    parser = argparse.ArgumentParser(description="network pytorch")
    parser.add_argument("--model", type=str, default="./pretrained_model/", help='checkpoint')
    parser.add_argument("--model_name", type=str, default='', help='model name')
    parser.add_argument("--test", type=str, default="./test_imgs/", help='input syn path')
    parser.add_argument("--output", type=str, default="./result/", help='output syn path')
    argspar = parser.parse_args()

    print("\nnetwork pytorch")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    arg = parser.parse_args()

    # train
    print('> Loading pretrained_model')
    name = arg.model_name
    model_name = name+'.tar'
    Model, _, _ = load_checkpoint(argspar.model, VRDNet, model_name)

    os.makedirs(arg.outest, exist_ok=True)
    test(argspar, Model)

def test(argspar, model):
    # init
    norm = lambda x: (x - 0.5) / 0.5
    denorm = lambda x: (x + 1) / 2
    files = os.listdir(argspar.intest)
    time_test = []
    model.eval()
    # test
    for i in range(len(files)):
        haze = np.array(Image.open(argspar.intest + files[i]).convert('RGB')) / 255
        
        with torch.no_grad():
            haze = torch.Tensor(haze.transpose(2, 0, 1)[np.newaxis, :, :, :]).cuda()
            haze = tensor2cuda(haze)

            starttime = time.time()
            haze = norm(haze)
            out = model(haze)
            endtime1 = time.time()
            
            out = denorm(out)
            imwrite(out, argspar.outest + files[i], range=(0, 1))
            
            time_test.append(endtime1 - starttime)

            print('The ' + str(i) + ' Time: %.4f s.' % (endtime1 - starttime))
    print('Mean Time: %.4f s.'%(sum(time_test)/len(time_test)))


if __name__ == '__main__':
    main()
