from .unet import custom_unet as unet
from segmentation_models import Unet as segmentations_unet
from tensorflow.keras.regularizers import l1_l2
from activations import get_activation
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model

def get_model(image_size, job_config):

    regularizer = l1_l2(l1=job_config["L1_REG"], l2=job_config["L2_REG"])
    model = None

    if job_config["MODEL"]["NAME"] == "unet":
        model = unet(
            input_shape=image_size,
            use_batch_norm=job_config["BATCH_NORM"],
            filters=job_config["MODEL"]["FILTERS"],
            dropout=job_config["DROPOUT"],
            dropout_change_per_layer=job_config["DROPOUT_CHANGE_PER_LAYER"],
            use_dropout_on_upsampling=job_config["USE_DROPOUT_ON_UPSAMPLE"],
            activation=get_activation(job_config["MODEL"]["ACTIVATION"]),
            kernel_initializer='he_normal',
            num_layers=job_config["MODEL"]["LAYERS"]
        )

    if job_config["MODEL"]["NAME"] == "segmentations_unet":
        base_model = segmentations_unet(
            backbone_name=job_config["MODEL"]["BACKBONE"]
        )
        inp = Input(shape=(None, None, 1))
        l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
        out = base_model(l1)
        model = Model(inp, out)

    if model is None:
        raise ValueError("Model %s not defined" % job_config["MODEL"]["NAME"])

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    return model