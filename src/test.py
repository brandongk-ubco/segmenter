from albumentations import (Blur, Compose, HorizontalFlip, HueSaturationValue,
                            IAAEmboss, IAASharpen, JpegCompression, OneOf,
                            RandomBrightness, RandomBrightnessContrast,
                            RandomContrast, RandomCrop, RandomGamma,
                            RandomRotate90, RGBShift, ShiftScaleRotate,
                            Transpose, VerticalFlip, ElasticTransform, GridDistortion, OpticalDistortion)
from augment import DataGeneratorFolder
from matplotlib import pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard, TerminateOnNaN, ProgbarLogger, BaseLogger, TerminateOnNaN, CSVLogger, LambdaCallback
from segmentation_models import Unet
from keras.optimizers import Adam
from segmentation_models.metrics import iou_score
from segmentation_models.losses import bce_jaccard_loss
from keras.layers import Input, Conv2D
from keras.models import Model
import os
import cv2

def aug_with_crop(image_size=(256,1600), crop_prob=1):
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
    ], p=1)


num_training_images = len(os.listdir('/data/one/fold0/train/images'))
num_val_images = len(os.listdir('/data/one/fold0/val/images'))


train_generator = DataGeneratorFolder(root_dir='/data/one/fold0/train',
                                     image_folder='images/',
                                     mask_folder='segmentations/',
                                     batch_size=1,
                                     image_size=(256,1600),
                                     image_divisibility=(32,32),
                                     channels=1,
                                     nb_y_features=1,
                                     augmentation=aug_with_crop)

val_generator = DataGeneratorFolder(root_dir='/data/one/fold0/val',
                                     image_folder='images/',
                                     mask_folder='segmentations/',
                                     batch_size=1,
                                     image_size=(256,1600),
                                     image_divisibility=(32,32),
                                     nb_y_features=1,
                                     channels=1,
                                     augmentation=None)

# os.makedirs("/output/augmented/train", exist_ok=True)
# for i in range(num_training_images):
#     images, masks = train_generator.__getitem__(i)
#     fig = plt.figure()
#     plt.tight_layout()
#     ax1 = fig.add_subplot(211)
#     ax2 = fig.add_subplot(212)
#     ax1.imshow(images[0, :, :, 0], cmap='gray')
#     ax2.imshow(masks[0, :, :, 0], cmap='gray')
#     plt.savefig("/output/augmented/train/%s.png" % i)
#     plt.close()

# images, masks = val_generator.__getitem__(0)

# os.makedirs("/output/augmented/val", exist_ok=True)
# for i in range(num_val_images):
#     images, masks = val_generator.__getitem__(i)
#     fig = plt.figure()
#     plt.tight_layout()
#     ax1 = fig.add_subplot(211)
#     ax2 = fig.add_subplot(212)
#     ax1.imshow(images[0, :, :, 0], cmap='gray')
#     ax2.imshow(masks[0, :, :, 0], cmap='gray')
#     plt.savefig("/output/augmented/val/%s.png" % i)
#     plt.close()

lr_reducer = ReduceLROnPlateau(factor=0.5,
                               cooldown=50,
                               patience=30, verbose=1,
                               min_lr=0.1e-7,
                               monitor='val_iou_score', mode='max')

mode_autosave = ModelCheckpoint("/output/best.h5", monitor='val_iou_score',
                                mode='max', save_best_only=True, verbose=1)

early_stopping = EarlyStopping(patience=100, verbose=1, monitor='val_iou_score', mode='max')

tensorboard = TensorBoard(log_dir='/output/logs/tenboard')

os.makedirs('/output/logs/one/', exist_ok=True)
logger = CSVLogger('/output/logs/one/fold0.csv', separator=',', append=False)

callbacks = [mode_autosave, lr_reducer, early_stopping, TerminateOnNaN(), logger]

unet_model = Unet(backbone_name='efficientnetb0', classes=1, encoder_weights='imagenet', encoder_freeze=False, input_shape=(256,1600,3), decoder_use_batchnorm=False)

inp = Input(shape=(256,1600,1))
l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
out = unet_model(l1)

model = Model(inp, out, name=unet_model.name)

model.compile(optimizer=Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, amsgrad=True),
              loss=bce_jaccard_loss, metrics=[iou_score])

history = model.fit_generator(train_generator, shuffle=True,
                              epochs=1000, workers=4, use_multiprocessing=True, steps_per_epoch=200,
                              validation_data=val_generator, validation_steps=200, callbacks=callbacks)
