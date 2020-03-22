'''
train
'''
from pathlib import Path
import torch
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils.model import get_efficient, get_seresnext
from utils.prepare_data import prepare_image_128

ROOT = Path("./data/")
SEED = 2020
indices=[0, 1, 2, 3]

def main():
    '''
    main
    '''
    train = pd.read_csv(f'{ROOT}/train.csv')
    train_labels_ = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values
    train_images_ = prepare_image_128(indices=indices)
    
    mskf = MultilabelStratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
    for train_index, test_index in mskf.split(train_images_, train_labels_):
        print("TRAIN:", train_index, "TEST:", test_index)
        train_images = train_images_[train_index]
        test_images = train_images_[test_index]
        train_labels = train_labels_[train_index]
        test_labels = train_labels_[test_index]

        model = get_efficient('efficientnet-b0', channels='GREY', pretrained=None)
        model_r = get_seresnext("se_resnext50_32x4d", channels='GREY', pretrained=None)
        inp = torch.empty(16, 1, 128, 128)
        #print(model(inp), model_r(inp))


if __name__ == '__main__':
    main()
