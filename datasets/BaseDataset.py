from math import sqrt
import numpy as np

class BaseDataset:

    # These need to be overriden by your base class.
    def get_classes(self):
        pass

    def get_class_counts(self):
        pass

    def get_class_members(self):
        pass

    def get_instances(self):
        pass

    def get_name(self, instance):
        pass

    def get_image(self, instance)
        pass

    def get_mask(self, instance):
        pass

    def enhance_image(self, image):
        pass
    #Done overriding.

    def chunkify(self, lst,n):
        return [lst[i::n] for i in range(n)]

    def split_folds(self, class_members, num_folds):
        chunked = {}
        for clazz, members in class_members.items():
            chunked[str(clazz)] = self.chunkify(members["train_instances"], num_folds)
        folds = []
        for fold_number in range(num_folds):
            fold = {}
            for clazz, members in chunked.items():
                fold[clazz] = {
                    "val": members[fold_number],
                    "train": sum([members[i] for i in range(len(members)) if i != fold_number], [])
                }
            folds.append(fold)

        return folds

    def process_images(self, process_lambda):
        _sum = 0
        _sum_2 = 0
        N = 0
        for instance in self.get_instances():
            mask = self.get_mask(instance)
            name = self.get_name(instance)
            image = self.get_image(instance)
            image = self.enhance_image(image)

            if np.max(mask) == 0:
                print("Mask has no values for %s" % name)
                continue
            print(name)

            process_lambda(image, mask, name)

            _sum += np.sum(image)
            _sum_2 += np.sum(np.square(image))
            N += image.size
        mean = _sum / N
        std = sqrt(_sum_2 / N - (_sum / N)**2)
        return mean, std