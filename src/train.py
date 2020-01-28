from albumentations import *
from augment import DataGeneratorFolder
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard, TerminateOnNaN, ProgbarLogger, BaseLogger, TerminateOnNaN, CSVLogger, LambdaCallback
from segmentation_models import Unet
from tensorflow.keras.optimizers import Adam
from segmentation_models.metrics import f1_score
from segmentation_models.losses import binary_focal_loss, dice_loss, binary_crossentropy, DiceLoss, BinaryFocalLoss
from segmentation_models.base import Loss
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
from models import unet
import os
import cv2
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import backend
import tensorflow as tf
from math import ceil

def aug_with_crop(image_size, crop_prob=1):
    box_scale = min(image_size)
    rescale_percentage = 0.5
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        OneOf([
            Compose([
                RandomScale(scale_limit=(0, rescale_percentage), p=1, interpolation=cv2.INTER_CUBIC),
                RandomCrop(image_size[0], image_size[1], p=1),
            ], p=0.5),
            RandomResizedCrop(image_size[0], image_size[1], scale=(1 - rescale_percentage, 1 + rescale_percentage), ratio=(1600/256, 1600/256), interpolation=cv2.INTER_CUBIC, p=0.5)
        ], p=1),
        ElasticTransform(p=0.5, alpha=box_scale, sigma=box_scale * 0.05, alpha_affine=box_scale * 0.03)
    ], p=1)


class NormalizedFocalLoss(Loss):

    def __call__(self, gt, pr, **kwargs):
        dl = dice_loss(gt, pr, **kwargs)
        bfl = BinaryFocalLoss(alpha=1, gamma=5)(gt, pr, **kwargs)
        bce = binary_crossentropy(gt, pr, **kwargs)

        ce_loss = bfl + bce
        
        return ce_loss + ce_loss * dl / backend.mean(ce_loss)

if __name__ == "__main__":

    for clazz in ["two", "one", "four", "three"]:
        for fold in range(5):

            print("Training class %s, fold %s" % (clazz, fold))

            train_folder = '/data/%s/fold%s/train/' % (clazz, fold)
            val_folder = '/data/%s/fold%s/val/' % (clazz, fold)
            output_folder = "/output/%s/fold%s/" % (clazz, fold)
            log_folder = os.path.join(output_folder, "logs")
            os.makedirs(output_folder, exist_ok=True)
            os.makedirs(log_folder, exist_ok=True)
            
            num_training_images = len(os.listdir(os.path.join(train_folder, "images")))
            num_val_images = len(os.listdir(os.path.join(val_folder, "images")))

            print("Found %s training images" % num_training_images)
            print("Found %s validation images" % num_val_images)

            BATCH_SIZE = 4

            train_generator = DataGeneratorFolder(root_dir=train_folder,
                                                image_folder='images/',
                                                mask_folder='segmentations/',
                                                batch_size=BATCH_SIZE,
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
                                        cooldown=20,
                                        patience=20, verbose=1,
                                        min_lr=1e-8,
                                        monitor='val_f1-score',
                                        mode='max',
                                        min_delta=1e-2)

            mode_autosave = ModelCheckpoint(os.path.join(output_folder, "weights.{epoch:02d}-{val_f1-score:.2f}.hdf5"), monitor='val_f1-score',
                                            mode='max', save_best_only=True, verbose=1)

            early_stopping = EarlyStopping(patience=60, verbose=1, monitor='val_f1-score', mode='max')

            tensorboard = TensorBoard(log_dir=os.path.join(log_folder, "tenboard"), profile_batch=0)

            logger = CSVLogger(os.path.join(log_folder, 'train.csv'), separator=',', append=False)
            
            callbacks = [mode_autosave, lr_reducer, early_stopping, TerminateOnNaN(), logger, tensorboard]

            model = unet(input_shape=(256,1600,1), use_batch_norm=True, filters=16, dropout=0.2, dropout_change_per_layer=0, use_dropout_on_upsampling=True)

            regularizer = l1_l2(l1=3e-5, l2=3e-5)

            for layer in model.layers:
                for attr in ['kernel_regularizer']:
                    if hasattr(layer, attr):
                        setattr(layer, attr, regularizer)

            # parallel_model = tf.keras.utils.multi_gpu_model(model, gpus=1, cpu_merge=True, cpu_relocation=True)

            model.compile(optimizer=Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, amsgrad=True),
                    loss=NormalizedFocalLoss(), metrics=[f1_score])

            history = model.fit(x=train_generator, validation_data=val_generator, shuffle=True,
                                        epochs=1000, steps_per_epoch=int(num_training_images/BATCH_SIZE),
                                        validation_steps=num_val_images, callbacks=callbacks, verbose=1)
