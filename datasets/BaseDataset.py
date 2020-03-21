from math import sqrt
import numpy as np

class BaseDataset:
    """
    A base class for creating datasets.  Your class should subclass this class.
    You may need to append this module into the system path before importing

    ```import sys
    sys.path.append("..")
    from BaseDataset import BaseDataset

    class MylDataset(BaseDataset)```
    """

    # These need to be overriden by your base class.
    def get_class_members(self):
        """
        Defines which members exist in which class.

        Should be a Python Dictionary of the format:
        ```
        {
            "classname_1": {
                "eval_instances": [
                    "eval_instance_1_name",
                    "eval_instance_2_name"
                ],
                "train_instances": [
                    "train_instance_1_name",
                    "train_instance_2_name"
                ]
            },
            "classname_2": {
                "eval_instances": [
                    "eval_instance_1_name",
                    "eval_instance_4_name"
                ],
                "train_instances": [
                    "train_instance_1_name",
                    "train_instance_4_name"
                ]
            }
        }
        ```
        :return: an dictionary containing which members are in which class, and which should be used for training or evaluation..
        """
        pass

    def iter_instances(self):
        """
        Creates a python generator that can be used to iterate over all instances in the dataset.
        Instances can be an arbitrary representation, and the instance will be passed to underlying functions
        get_name, get_image, and get_mask for processing.

        :return: a python generator that iterates over each instance.
        """
        pass

    def get_name(self, instance):
        """
        Converts a representation produced by iter_instances into the representative name.

        Generally, this will be the name the file is saved with on disk, with extension .npy added.
        As such, this name should not have an extension.

        :instance: The instance representation as produced by iter_instances
        :return: a String of instance name.
        """
        pass

    def get_image(self, instance):
        """
        Given an instance produced by iter_instances, loads the image into a numpy array.

        :instance: The instance representation as produced by iter_instances
        :return: a numpy array of the image.
        """
        pass

    def get_masks(self, instance):
        """
        Given an instance produced by iter_instances, load the mask into a numpy array.

        The mask should be n channels, where n matches the length of the get_classes function.
        Each channel should be of the same shape as the image produced by enhance_image.

        :instance: The instance representation as produced by iter_instances
        :return: a numpy array of the masks.
        """
        pass

    def enhance_image(self, image):
        """
        Given an image, performs and enhancement or processing required and returns the result.

        Examples of processing here can be contrast stretching, histogram equalization, or CLAHE.

        The mean and stard deviation of the dataset will be calculated after this processing, and the processed
        images will be what is saved to dist.

        :image: np.ndarray of the image to be enhanced.
        :return: np.ndarray of the enhanced image.
        """
        pass

    #Done overriding.

    def chunkify(self, lst,n):
        """
        Given a list and a number of chunks, returns a list of lists of chunks.
        Chunks will be split as evenly as possible.
        Each member in the original list will be represented in only one chunk.

        :lst: List to be chunked
        :n: Integer number of chunks
        :return: a list of list of chunks
        """
        return [lst[i::n] for i in range(n)]

    def split_folds(self, class_members, num_folds):
        """
        Given a dictionary of class members returned by get_class_members and a number of folds,
        returns a list of folds, where each list item is a dictionary of the fold definition.

        :class_members: Dictionary as produced by get_class_members
        :num_folds: Integer
        :return: List of folds, where each list item is a dictionary of the fold definition.
        """
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
        """
        Iterates over all instances from iter_instances, collects the name, mask, and enhanced image,
        and class the process_lambda passed in.

        :process_lambda: Lambda function which accepts arguments (image: np.ndarray, mask: np.ndarray, name: String)
          and should persist the file to disk in the correct location.
        :return: Tuple of (mean: float, std: float) of the mean and standard deviation over the dataset.
        """
        _sum = 0
        _sum_2 = 0
        N = 0
        for instance in self.iter_instances():
            masks = self.get_masks(instance)
            name = self.get_name(instance)
            image = self.get_image(instance)
            image = self.enhance_image(image)

            if np.max(mask) == 0:
                print("Mask has no values for %s" % name)
                continue
            print(name)

            process_lambda(image, masks, name)

            _sum += np.sum(image)
            _sum_2 += np.sum(np.square(image))
            N += image.size
        mean = _sum / N
        std = sqrt(_sum_2 / N - (_sum / N)**2)
        return mean, std