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
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import backend
import tensorflow as tf
from math import ceil
from data import DataGenerator
from augmentations import augment

class NormalizedFocalLoss(Loss):

    def __call__(self, gt, pr, **kwargs):
        dl = dice_loss(gt, pr, **kwargs)
        bfl = BinaryFocalLoss(alpha=1, gamma=5)(gt, pr, **kwargs)
        bce = binary_crossentropy(gt, pr, **kwargs)

        ce_loss = bfl + bce
        
        return ce_loss + ce_loss * dl / backend.mean(ce_loss)

if __name__ == "__main__":

    for clazz in ["2", "1", "4", "3"]:
        for fold in range(5):

            print("Training class %s, fold %s" % (clazz, fold))

            train_folder = '/data/%s/fold%s/train/' % (clazz, fold)
            val_folder = '/data/%s/fold%s/val/' % (clazz, fold)
            output_folder = "/output/%s/fold%s/" % (clazz, fold)
            log_folder = os.path.join(output_folder, "logs")
            os.makedirs(output_folder, exist_ok=True)
            os.makedirs(log_folder, exist_ok=True)

            BATCH_SIZE = 4

            train_generator = DataGenerator(clazz, fold, augmentations=augment)
            train_dataset = tf.data.Dataset.from_generator(train_generator.generate, (tf.float32,tf.float32),
                output_shapes=(tf.TensorShape((None, None, None)), tf.TensorShape((None, None, None))))
            train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)

            val_generator = DataGenerator(clazz, fold, mode="val")
            val_dataset = tf.data.Dataset.from_generator(val_generator.generate, (tf.float32,tf.float32),
                output_shapes=(tf.TensorShape((256, 1600, 1)), tf.TensorShape((256, 1600, 1))))
            val_dataset = val_dataset.batch(1, drop_remainder=False)

            num_training_images = train_generator.size()
            num_val_images = val_generator.size()

            print("Found %s training images" % num_training_images)
            print("Found %s validation images" % num_val_images)

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

            train_shuffler = LambdaCallback(on_epoch_end= lambda epoch, logs: train_generator.shuffle())
            
            callbacks = [mode_autosave, lr_reducer, early_stopping, TerminateOnNaN(), logger, tensorboard, train_shuffler]

            model = unet(input_shape=(256,1600,1), use_batch_norm=True, filters=16, dropout=0.2, dropout_change_per_layer=0, use_dropout_on_upsampling=True)

            regularizer = l1_l2(l1=3e-5, l2=3e-5)

            for layer in model.layers:
                for attr in ['kernel_regularizer']:
                    if hasattr(layer, attr):
                        setattr(layer, attr, regularizer)

            # parallel_model = tf.keras.utils.multi_gpu_model(model, gpus=1, cpu_merge=True, cpu_relocation=True)

            model.compile(optimizer=Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, amsgrad=True),
                    loss=NormalizedFocalLoss(), metrics=[f1_score])

            history = model.fit(x=train_dataset, validation_data=val_dataset,
                                        epochs=1000, steps_per_epoch=int(num_training_images/BATCH_SIZE),
                                        validation_steps=num_val_images, callbacks=callbacks, verbose=1)
