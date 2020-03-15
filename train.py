'''
train
'''

import torch
from utils.model import get_efficient, get_seresnext


def main():
    '''
    main
    '''
    model = get_efficient('efficientnet-b0', channels='GREY')
    model_r = get_seresnext("se_resnext50_32x4d", channels='GREY')
    inp = torch.empty(16, 1, 128, 128)
    print(model(inp), model_r(inp), inp)


if __name__ == '__main__':
    main()
