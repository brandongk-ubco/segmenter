from models import unet
import os
from skimage.io import imread
import numpy as np
import random

def find_best_weight(folder):
    return max([os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))], key=os.path.getctime)

if __name__ == "__main__":

    included = [f[:-4] for f in os.listdir("/data/masks")]

    models = []
    predictions_folder = "/output/predictions/"
    os.makedirs(predictions_folder, exist_ok=True)

    for clazz in ["two", "one", "four", "three"]:
        for fold in os.listdir("/output/%s" % clazz):
            model = unet(input_shape=(256,1600,1), use_batch_norm=True, filters=16, dropout=0.2, dropout_change_per_layer=0, use_dropout_on_upsampling=True)
            weights_folder = "/output/%s/%s/" % (clazz, fold)
            weight_file = find_best_weight(weights_folder)
            model.load_weights(weight_file)
            models.append(model)

    images_folder = "/data/images/"
    images = [(f, os.path.join(images_folder, f)) for f in os.listdir(images_folder) if f[:-4] in included][:1000]
    random.shuffle(images)
    for image_name, image_path in images:
        image = imread(image_path, as_gray=True)
        to_predict = np.empty((1, image.shape[0], image.shape[1], 1), dtype=np.float32)
        predictions = np.empty((image.shape[0], image.shape[1], len(models)*4), dtype=np.float32)
        print(image_name[:-4])
        for j, img in enumerate([image, np.fliplr(image), np.flipud(image), np.fliplr(np.flipud(image))]):
            for i, model in enumerate(models):
                to_predict[0, :, :, 0] = img
                if j == 1:
                    prediction = np.fliplr(prediction)
                if j == 2:
                    prediction = np.flipud(prediction)
                if j == 3:
                    prediction = np.fliplr(np.flipud(prediction))
                prediction = model.predict(to_predict)[0, :, :, 0]
                predictions[:,:,i+j*len(models)] = prediction
        np.savez_compressed("%s/%s" % (predictions_folder, image_name[:-4]), predictions=predictions)
