import numpy as np
import albumentations as albu
from albumentations.pytorch import ToTensor
from albumentations.pytorch import ToTensorV2

BORDER_CONSTANT = 0
BORDER_REFLECT = 2
RESCALE_SIZE = 224

def pre_transforms(image_size=RESCALE_SIZE):
    # Convert the image to a square of size image_size x image_size
    # (keeping aspect ratio)
    result = [
        albu.LongestMaxSize(max_size=image_size),
        albu.PadIfNeeded(image_size, image_size, border_mode=BORDER_CONSTANT)
    ]
    
    return result

def hard_transforms():
    result = [
                #albu.JpegCompression(quality_lower=20, quality_upper=40, p=.1),
                albu.ShiftScaleRotate(p=.5, scale_limit=(-.2, .2), rotate_limit=5, shift_limit=(-.01,.01), border_mode=0),
                albu.GridDistortion(p=.5, border_mode=0, distort_limit=.25),

                #albu.ElasticTransform(p=.1, alpha=70, sigma=9, alpha_affine=7),
                # albu.OneOf([
                #albu.CoarseDropout(max_holes=8, max_height=32, max_width=16, min_holes=4, min_height=2, min_width=2, fill_value=0, p=.1),
                albu.GridDropout(ratio=.5, p=.5),
                #], p=.3),
                # albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                # #albu.CenterCrop(112, 112, .3),
                albu.Blur(blur_limit=3, p=.3),              
    ]
   
    return result

class ToTensorV3(ToTensorV2):
    def apply(self, img, **params):
        return torch.from_numpy(np.expand_dims(img, axis=2).transpose(2, 0, 1))

def post_transforms(image_size=RESCALE_SIZE):
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [
            # albu.LongestMaxSize(max_size=image_size),
            # albu.PadIfNeeded(image_size, image_size, border_mode=BORDER_CONSTANT),
            albu.Normalize(mean=[np.mean([0.485, 0.456, 0.406])], std=[np.mean([0.229, 0.224, 0.225])]),  #mean=0.069, std=0.205),  #
            ToTensorV3()
            ]

def compose(transforms_to_compose):
    # combine all augmentations into one single pipeline
    result = albu.Compose([
      item for sublist in transforms_to_compose for item in sublist
    ])
    return result

train_transforms = compose([
    hard_transforms(), 
    post_transforms(),
])
valid_transforms = compose([post_transforms()])

show_transforms = compose([hard_transforms()])