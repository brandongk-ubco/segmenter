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
from segmentation_models.metrics import f1_score
from segmentation_models.losses import binary_focal_loss, dice_loss, binary_crossentropy, DiceLoss, BinaryFocalLoss
from segmentation_models.base import Loss
from keras.layers import Input, Conv2D
from keras.models import Model
from models import unet
import os
import cv2
from keras.regularizers import l1_l2
from keras.backend import mean, shape

def aug_with_crop(image_size=(256,1600), crop_prob=1):
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
    ], p=1)

class NormalizedFocalLoss(Loss):

    def __call__(self, gt, pr, **kwargs):
        dl = dice_loss(gt, pr, **kwargs)
        bfl = BinaryFocalLoss(alpha=1, gamma=5)(gt, pr, **kwargs)
        bce = binary_crossentropy(gt, pr, **kwargs)

        ce_loss = bfl + bce

        return ce_loss * dl / mean(ce_loss)

for fold in [1,2,3,4]:

    print("Training fold %s" % fold)

    train_folder = '/data/one/fold%s/train' % fold
    val_folder = '/data/one/fold%s/val' % fold
    output_folder = "/output/one/fold%s/" % fold
    log_folder = os.path.join(output_folder, "logs")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)
    
    num_training_images = len(os.listdir(os.path.join(train_folder, "images")))
    num_val_images = len(os.listdir(os.path.join(val_folder, "images")))


    BATCH_SIZE = 4

    train_generator = DataGeneratorFolder(root_dir=train_folder,
                                        image_folder='images/',
                                        mask_folder='segmentations/',
                                        batch_size=4,
                                        image_size=(256,1600),
                                        image_divisibility=(32,32),
                                        channels=1,
                                        nb_y_features=1,
                                        augmentation=aug_with_crop)

    val_generator = DataGeneratorFolder(root_dir=val_folder,
                                        image_folder='images/',
                                        mask_folder='segmentations/',
                                        batch_size=1,
                                        image_size=(256,1600),
                                        image_divisibility=(32,32),
                                        nb_y_features=1,
                                        channels=1,
                                        augmentation=None)

    lr_reducer = ReduceLROnPlateau(factor=0.1,
                                cooldown=10,
                                patience=10, verbose=1,
                                min_lr=1e-8,
                                monitor='val_f1-score',
                                mode='max',
                                min_delta=1e-2)

    mode_autosave = ModelCheckpoint(os.path.join(output_folder, "/weights.{epoch:02d}-{val_f1-score:.2f}.hdf5"), monitor='val_f1-score',
                                    mode='max', save_best_only=True, verbose=1)

    early_stopping = EarlyStopping(patience=30, verbose=1, monitor='val_f1-score', mode='max')

    # tensorboard = TensorBoard(log_dir=os.path.join(log_folder, "tenboard"))

    logger = CSVLogger(os.path.join(log_folder, 'train.csv'), separator=',', append=False)
    
    callbacks = [mode_autosave, lr_reducer, early_stopping, TerminateOnNaN(), logger]

    # unet_model = Unet(backbone_name='efficientnetb0', classes=1, encoder_weights='imagenet', encoder_freeze=False, input_shape=(256,1600,3), decoder_use_batchnorm=False)
    # inp = Input(shape=(256,1600,1))
    # l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
    # out = unet_model(l1)
    # model = Model(inp, out, name=unet_model.name)

    model = unet(input_shape=(256,1600,1), use_batch_norm=True, filters=16, dropout=0.3, dropout_change_per_layer=0.05, use_dropout_on_upsampling=True)

    regularizer = l1_l2(l1=3e-5, l2=3e-5)

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)


    model.compile(optimizer=Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, amsgrad=True),
                loss=NormalizedFocalLoss(), metrics=[f1_score])

    history = model.fit_generator(train_generator, shuffle=True,
                                epochs=1000, workers=BATCH_SIZE, use_multiprocessing=True, steps_per_epoch=int(num_training_images/BATCH_SIZE),
                                validation_data=val_generator, validation_steps=num_val_images, callbacks=callbacks)
