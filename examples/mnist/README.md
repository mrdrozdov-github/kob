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


Environment Setup

```
export LD_LIBRARY_PATH=/Users/Andrew/Developer/kob-setup/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/Users/Andrew/anaconda/lib:$LD_LIBRARY_PATH

git clone git@github.com:gflags/gflags.git
cd gflags
mkdir build_cmake
cd build_cmake
cmake -D CMAKE_BUILD_TYPE=Release ..
make
make install
```
