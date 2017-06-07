#include "batch_reader.h"

BatchReader::BatchReader(string filename, string datasetname, int n, int size) {
    this->rank = 2;

    this->filename = filename;
    this->datasetname = datasetname;
    this->n = n;
    this->size = size;
}

void BatchReader::read_item(float *out, int index) {
    H5File file(this->filename, H5F_ACC_RDONLY);
    DataSet dataset = file.openDataSet(this->datasetname);

    hsize_t offset[2], count[2], stride[2], block[2];
    hsize_t dims[2], dimsm[2];

    // Size of dataset.
    dims[0] = this->n;
    dims[1] = this->size;

    // Size to read.
    dimsm[0] = 1;
    dimsm[1] = this->size;

    offset[0] = index;
    offset[1] = 0;

    count[0] = 1;
    count[1] = this->size;

    stride[0] = 1;
    stride[1] = 1;

    block[0] = 1;
    block[1] = 1;

    DataSpace dataspace = DataSpace(this->rank, dims);
    DataSpace memspace(this->rank, dimsm, NULL);

    // Read data.
    dataspace.selectHyperslab(H5S_SELECT_SET, count, offset, stride, block); 
    dataset.read(out, PredType::NATIVE_FLOAT, memspace, dataspace);

    dataspace.close();
    memspace.close();
    dataset.close();
    file.close();
}

BatchReader::~BatchReader() {

}
