import h5py
import numpy as np
from datasets import MnistDataset

d = MnistDataset()

for section in ['train', 'validation', 'test']:
    dset = getattr(d, section)
    f = h5py.File('{}.h5'.format(section), 'w')
    f.create_dataset("images", data=dset.images)
    f.create_dataset("labels", data=dset.labels)
    f.close()
