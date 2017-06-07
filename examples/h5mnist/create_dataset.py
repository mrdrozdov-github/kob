import h5py
import numpy as np
from datasets import MnistDataset

d = MnistDataset()

images = np.concatenate([d.train.images, d.validation.images], axis=0)
labels = np.concatenate([d.train.labels, d.validation.labels], axis=0)
f = h5py.File('train.h5', 'w')
f.create_dataset("images", data=images)
f.create_dataset("labels", data=labels)
f.close()

dset = d.test
images = d.test.images
labels = d.test.labels
f = h5py.File('test.h5', 'w')
f.create_dataset("images", data=images)
f.create_dataset("labels", data=labels)
f.close()
