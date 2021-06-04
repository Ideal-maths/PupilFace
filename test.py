#--------------------------------------------#
#   The code shows the structure of networks,and it is not test code
#--------------------------------------------#
import torch
from torchsummary import summary

from nets.retinaface import RetinaFace
from utils.config import cfg_mnet

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RetinaFace(cfg_mnet).to(device)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    summary(model, input_size=(3, 2150, 2150))