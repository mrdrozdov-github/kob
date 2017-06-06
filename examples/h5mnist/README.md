Install HDF5:

https://www.hdfgroup.org/downloads/hdf5/


Export library path so linker doesn't bail:

```
export LD_LIBRARY_PATH=/Users/Andrew/Developer/myhdf5stuff/dist/HDF5-1.10.1-Darwin/HDF_Group/HDF5/1.10.1/lib:$LD_LIBRARY_PATH
```


Create mnist dataset:

```
python create_dataset.py
```


Example of reading dataset:

```
mkdir build
make
./build/demo
```