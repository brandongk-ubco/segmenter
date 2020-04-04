# Expected Format

The dataset is expected to consist of:

1. Many `.npz` files, each file being a complete training example.  The .npz file will have the original image and a one-hot-coded mask of classes.
2. Many `N-folds.json` files, which define how the classes should be divided in each fold.
3. A `classes.json` file, which defines how each class is divided between training / evaluation examples.

# Processing

The dataset module should export the dataset processor as `Dataset`.  This should be done in the `__init__.py` file.

The dataset processor should subclass the `BaseDataset.py` file.

```
import sys
sys.path.append("..")

from BaseDataset import BaseDataset

class MyDataset(BaseDataset):
    ...
```

There are functions which need to be overridden in your dataset.  They are described using inline documentation in `BaseDataset.py`
