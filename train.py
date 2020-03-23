'''
train
'''
from pathlib import Path
import torch
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from catalyst.dl.callbacks import MixupCallback
from utils.model import get_efficient, get_seresnext, get_lr_seresnext
from utils.prepare_data import prepare_image_128
from utils.albu import train_transforms, valid_transforms
from utils.dataset import BengaliAIDataset
from utils.settings import settings
from utils.loss import Loss_combine


N_FOLDS = 5
ROOT = Path("./data/")
SEED = 2020
indices = [0, 1, 2, 3]
NUM_CORES, BS = settings()
BS = 256

num_epochs = 50

def main():
    '''
    main
    '''
    train = pd.read_csv(f'{ROOT}/train.csv')
    train_labels_ = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values
    train_images_ = prepare_image_128(indices=indices)

    
    mskf = MultilabelStratifiedKFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)
    fold = 0
    for train_index, test_index in mskf.split(train_images_, train_labels_):
        print("TRAIN:", train_index, "TEST:", test_index)
        train_images = train_images_[train_index]
        test_images = train_images_[test_index]
        train_labels = train_labels_[train_index]
        test_labels = train_labels_[test_index]

        train_dataset = BengaliAIDataset(train_images, train_labels, transform=train_transforms, is_font=True)
        valid_dataset = BengaliAIDataset(test_images, test_labels, transform=valid_transforms, is_font=True)
        
        train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=NUM_CORES)
        valid_loader = DataLoader(valid_dataset, batch_size=BS, shuffle=False, num_workers=NUM_CORES)
        loaders = collections.OrderedDict()
        loaders["train"] = train_loader
        loaders["valid"] = valid_loader

        
        optimizer = torch.optim.AdamW(LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=2)
        #model = get_efficient('efficientnet-b0', channels='GREY', pretrained=None)
        model = get_seresnext("se_resnext50_32x4d", channels='GREY')
        LR = get_lr_seresnext(0.01, 0.8)
        logdir = f"{ROOT}/.logs{fold}"
        device = utils.get_device()

        runner = SupervisedRunner(device=device, model=model)
        from catalyst.dl.callbacks import AccuracyCallback, AUCCallback, F1ScoreCallback

        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            callbacks=[
                    BengaliRecall(prefix='total', average='macro'),
                    #MixupCallback(alpha=0.4)
                    ],
            logdir=logdir,
            
            num_epochs=num_epochs,
            main_metric="total",
            minimize_metric=False,
            # for FP16. It uses the variable from the very first cell
            #fp16=fp16_params,
            # for external monitoring tools, like Alchemy
            #monitoring_params=monitoring_params,
            
            # prints train logs
            verbose=True
        )
        fold += 1


if __name__ == '__main__':
    main()
