#include <string>
#include "H5Cpp.h"

using namespace std;
using namespace H5;

class BatchReader;

class BatchReader
{
    private:
        void dataset_read(const DataSet &dataset, float *out, const DataSpace &memspace, const DataSpace &dataspace);
    public:
        int rank, n, size;
        H5std_string filename, datasetname;
        void read_item(float *out, int index);
        void read_item(float *out, int index, const H5File &file, const DataSet &dataset);
        void read_batch(float *out, int *index, int batch_size);
        BatchReader(string filename, string datasetname, int n, int size);
        ~BatchReader();
};
