#include <math.h>
#include <iostream>

#include "../../kob/kob.h"
#include "../../kob/batch_reader.h"
#include "H5Cpp.h"
#include "gflags/gflags.h"

#include "json.hpp"

#define PRINT_SAMPLE false

#define SPACE1_DIM1 9815
#define SPACE1_RANK 1

using namespace std;
using json = nlohmann::json;

/*

TODO:

*/

DEFINE_string(train_file, "nli.h5", "Data file");

int main(int argc, char *argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    gflags::ShutDownCommandLineFlags();


    H5File file(FLAGS_train_file, H5F_ACC_RDONLY);
    DataSet dataset = file.openDataSet("sentence1_tokens");

    /* Get datatype for dataset */
    DataType dtype = dataset.getDataType();

    char *rdata[SPACE1_DIM1];

    printf("Reading...\n");

    /* Read dataset from disk */
    dataset.read((void*)rdata, dtype);

    /* Validate and print data read in */
    cout << "data read:" << endl;
    for(unsigned i=0; i<SPACE1_DIM1; i++) {
        cout << rdata[i] << endl;
        cout << rdata[i][0] << endl;
        cout << rdata[i][1] << endl;
        cout << rdata[i][2] << endl;
        string sample = rdata[i];
        cout << sample << endl;
        cout << endl;
        break;
    }
    for(unsigned i=0; i<SPACE1_DIM1; i++) {
        printf("Deserializing\n");
        string sample = rdata[i];
        auto j3 = json::parse(sample);
        printf("Printing\n");
        cout << j3.dump() << endl;
        cout << j3[0] << endl;
        string j3_sample = j3[0];
        cout << j3_sample << endl;
        cout << j3_sample[0] << endl;
        cout << j3[1] << endl;
        cout << j3[2] << endl;
        cout << endl;
        break;
    }

    /* Free memory for rdata */
    for(unsigned i=0; i<SPACE1_DIM1; i++) {
        free(rdata[i]);
    }

    dataset.close();
    file.close();

    return 0;
}
