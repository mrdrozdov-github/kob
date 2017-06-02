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
