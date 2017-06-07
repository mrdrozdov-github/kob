#include <string>
#include "H5Cpp.h"

using namespace std;
using namespace H5;

class BatchReader;

class BatchReader
{
    private:
    public:
        int rank, n, size;
        H5std_string filename, datasetname;
        void read_item(float *out, int index);
        void read_batch(float *out, int *index, int batch_size);
        BatchReader(string filename, string datasetname, int n, int size);
        ~BatchReader();
};
