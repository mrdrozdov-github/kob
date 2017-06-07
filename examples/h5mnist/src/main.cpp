/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Board of Trustees of the University of Illinois.         *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of HDF5.  The full HDF5 copyright notice, including     *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the root of the source code       *
 * distribution tree, or in https://support.hdfgroup.org/ftp/HDF5/releases.  *
 * If you do not have access to either file, you may request a copy from     *
 * help@hdfgroup.org.                                                        *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
//
//      This example reads hyperslab from the SDS.h5 file into
//      two-dimensional plane of a three-dimensional array.  Various
//      information about the dataset in the SDS.h5 file is obtained.
//
//      Modified version of: https://support.hdfgroup.org/HDF5/doc/cpplus_RM/examples.html
//
#ifdef OLD_HEADER_FILENAME
#include <iostream.h>
#else
#include <iostream>
#endif
using std::cout;
using std::endl;
#include <string>
#include <math.h>
#include "H5Cpp.h"
using namespace H5;
const H5std_string FILE_NAME("train.h5");
const H5std_string DATASET_NAME("images");
const int NX = 55000;
const int NY = 784;
const int _NX = 28;
const int _NY = 28;
const int _NZ = 1;
const int   RANK = 2;
int main (void)
{
  int i, j, k;
  float data_out[_NZ][NY];

  // Initialize output buffer.
  for (k = 0; k < _NZ; k++)
  {
    for (j = 0; j < _NY; j++)
    {
      for (i = 0; i < _NX; i++)
      {
        data_out[k*NY][j*_NY + i] = 0;
        cout << data_out[k*NY][j*_NY + i] << " ";
      }
      cout << endl;
    }
  }
  cout << endl;

  // Open file.
  H5File file(FILE_NAME, H5F_ACC_RDONLY);
  DataSet dataset = file.openDataSet(DATASET_NAME);

  // Read subset of data.
  hsize_t offset[2], count[2], stride[2], block[2];
  hsize_t dims[2], dimsm[2];

  offset[0] = 1; // TODO: Change this to access different elements in the array!
  offset[1] = 0;

  int NUM_BLOCKS = 1;
  int DIM0_SUB = 1;
  int DIM1_SUB = 784;

  dims[0] = 55000; // Num images.
  dims[1] = 784; // Size of image.

  count[0]  = DIM0_SUB;
  count[1]  = DIM1_SUB;

  stride[0] = 1;
  stride[1] = 1;

  block[0] = NUM_BLOCKS;
  block[1] = 1;

  // Define Memory Dataspace. Get file dataspace and select
  // a subset from the file dataspace.

  dimsm[0] = DIM0_SUB;
  dimsm[1] = DIM1_SUB;

  DataSpace dataspace = DataSpace(RANK, dims);
  DataSpace memspace(RANK, dimsm, NULL);

  dataspace.selectHyperslab(H5S_SELECT_SET, count, offset, stride, block); 

  // Read data.
  dataset.read(data_out, PredType::NATIVE_FLOAT, memspace, dataspace);

  // Print sample.
  for (k = 0; k < _NZ; k++)
  {
    for (j = 0; j < _NY; j++)
    {
      for (i = 0; i < _NX; i++)
      {
        cout << ceil(data_out[k*NY][j*_NY + i]) << " ";
      }
      cout << endl;
    }
  }
  /*
   * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
   * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
   * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
   * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
   * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
   * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
   * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
   * 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0
   * 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0
   * 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
   * 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0
   * 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0
   * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0
   * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0
   * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
   * 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
   * 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
   * 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
   * 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 0 0
   * 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0
   * 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0
   * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0
   * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0
   * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0
   * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0
   * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0
   * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0
   * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
   */

  return 0;
}
