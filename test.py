import argparse
import numpy as np
import torch
from torch.autograd import Variable
from model.deeplab_multi import DeeplabMulti
from model.discriminator import FCDiscriminator
from model.discriminator import OutspaceDiscriminator,Discriminator_aux
from dataset.test_dataset import TestDataSet
from torch.utils import data
from PIL import Image
import torch.nn as nn
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DATA_DIRECTORY = './data/testB/'
DATA_LIST_PATH = './dataset/ct_list/test.txt'
SAVE_PATH = './dice_score'
NUM_CLASSES = 5
NUM_STEPS = 1024 # Number of images in the test set.


def create_map(input_size, mode):
    if mode == 'h':
        T_base = torch.arange(0, float(input_size[1]))
        T_base = T_base.view(input_size[1], 1)
        T = T_base
        for i in range(input_size[0] - 1):
            T = torch.cat((T, T_base), 1)
        T = torch.div(T, float(input_size[1]))
    if mode == 'w':
        T_base = torch.arange(0, float(input_size[0]))
        T_base = T_base.view(1, input_size[0])
        T = T_base
        for i in range(input_size[1] - 1):
            T = torch.cat((T, T_base), 0)
        T = torch.div(T, float(input_size[0]))
    T = T.view(1, 1, T.size(0), T.size(1))
    return T


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--gpu", type=int, default='5',
                        help="choose gpu device.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()


def main():
        model_path = './pretrained_model/mr2ct.pth'
        model_D_path = './pretrained_model/mr2ct_D.pth'
        save_path = './result'
        args = get_arguments()
    
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        model = DeeplabMulti(num_classes=args.num_classes)
        saved_state_dict = torch.load(model_path)
        model.load_state_dict(saved_state_dict)
        model.eval()
        model.cuda()
        num_class_list = [2048, 5]
        model_D = nn.ModuleList()
        model_D.append(FCDiscriminator(num_classes=num_class_list[0]))
        model_D.append(OutspaceDiscriminator(num_classes=num_class_list[1]))
        model_D.load_state_dict(torch.load(model_D_path))
        model_D.eval()
        model_D.cuda()
            
        testloader = data.DataLoader(TestDataSet(args.data_dir, args.data_list, crop_size=(256,256), scale=False, mirror=False),
                                        batch_size=1, shuffle=False, pin_memory=True)
    
        interp = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
    
        with torch.no_grad():
            for index, batch in enumerate(testloader):
                if index % 100 == 0:
                    print('%d processd' % index)
                image,labels, _, name = batch
                feature, pred  = model(Variable(image).cuda(), model_D, 'target')
                output = interp(pred).cpu()
                name = name[0].split('/')[-1]
                output = (output.data[0].numpy()).transpose(1, 2, 0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
                output = output * 51
                output = Image.fromarray(output)
                output.save('%s/%s' % (save_path, name))

if __name__ == '__main__':
    main()
