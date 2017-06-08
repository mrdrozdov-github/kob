#include <string>
#include "H5Cpp.h"

using namespace std;
using namespace H5;

class BatchReader;

class BatchReader
{
    private:
        // float
        void dataset_read(const DataSet &dataset, float *out, const DataSpace &memspace, const DataSpace &dataspace);
        // int
        void dataset_read(const DataSet &dataset, long *out, const DataSpace &memspace, const DataSpace &dataspace);
    public:
        int rank, n, size;
        H5std_string filename, datasetname;

        // float
        void read_item(float *out, int index);
        void read_item(float *out, int index, const H5File &file, const DataSet &dataset);
        void read_batch(float *out, int *index, int batch_size);
        void read_batch(float *out, int *index, int batch_size, const H5File &file, const DataSet &dataset);
        // int
        void read_item(long *out, int index);
        void read_item(long *out, int index, const H5File &file, const DataSet &dataset);
        void read_batch(long *out, int *index, int batch_size);
        void read_batch(long *out, int *index, int batch_size, const H5File &file, const DataSet &dataset);

        BatchReader(string filename, string datasetname, int n, int size);
        ~BatchReader();
};
