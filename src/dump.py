from train import aug_with_crop
from augment import DataGeneratorFolder
from matplotlib import pyplot as plt
import os

for clazz in ["two"]:
    for fold in [0]:

        print("Dumping augments for class %s, fold %s" % (clazz, fold))
        output_folder = "/output/%s/fold%s/augmented" % (clazz, fold)
        train_folder = '/data/%s/fold%s/train/' % (clazz, fold)
        num_training_images = len(os.listdir(os.path.join(train_folder, "images")))

        train_generator = DataGeneratorFolder(root_dir=train_folder,
                                            image_folder='images/',
                                            mask_folder='segmentations/',
                                            batch_size=1,
                                            image_size=(256,1600),
                                            image_divisibility=(32,32),
                                            channels=1,
                                            nb_y_features=1,
                                            augmentation=aug_with_crop)

        os.makedirs(output_folder, exist_ok=True)
        for i in range(num_training_images):
            print(i)
            images, masks = train_generator.__getitem__(i)
            fig = plt.figure()
            plt.tight_layout()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            ax1.imshow(images[0, :, :, 0], cmap='gray')
            ax2.imshow(masks[0, :, :, 0], cmap='gray')
            plt.savefig(os.path.join(output_folder, "%s.png" % i))
            plt.close()
