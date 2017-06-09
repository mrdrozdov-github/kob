#include <math.h>
#include <iostream>
#include <map>

#include "../../kob/kob.h"
#include "../../kob/batch_reader.h"
#include "../../kob/logging.h"
#include "H5Cpp.h"
#include "gflags/gflags.h"

#include "json.hpp"


#define SPACE1_RANK 1

using namespace std;
using json = nlohmann::json;

/*

TODO:

*/

DEFINE_string(input_file, "train.h5", "Data file");

void read_variable_length_data(H5File &file, DataSet &dataset, void *out, int _offset)
{
    /* Get datatype for dataset */
    DataType dtype = dataset.getDataType();

    hsize_t offset[1], count[1], stride[1], block[1];
    hsize_t dims[1], dimsm[1];

    int num_tokens = dataset.getSpace().getSelectNpoints();
    dims[0] = num_tokens;
    dimsm[0] = 1;

    offset[0] = _offset; // NOTE: Change this to select different elements in the dataset.
    count[0] = 1;
    stride[0] = 1;
    block[0] = 1;

    DataSpace dataspace = DataSpace(SPACE1_RANK, dims);
    DataSpace memspace(SPACE1_RANK, dimsm, NULL);

    /* Read dataset from disk */
    dataspace.selectHyperslab(H5S_SELECT_SET, count, offset, stride, block); 
    dataset.read(out, dtype, memspace, dataspace);
}

void dataset_example(H5File &file)
{
    DataSet dataset = file.openDataSet("sentence1_tokens");
    int num_tokens = dataset.getSpace().getSelectNpoints();
    printf("%d\n", num_tokens);
}

void tokens_example(H5File &file)
{
    int offset = 13;

    // Tokens example.
    DataSet tokens_dataset = file.openDataSet("sentence1_tokens");
    char *tokens_data[1];
    read_variable_length_data(file, tokens_dataset, tokens_data, offset);

    string tokens_string = tokens_data[0];
    auto tokens_json = json::parse(tokens_string);
    cout << tokens_json << endl;
    string tokens_sample = tokens_json[0];
    cout << tokens_sample << endl;

    free(tokens_data[0]);
    tokens_dataset.close();
}

void transitions_example(H5File &file)
{
    int offset = 13;

    // Transitions example.
    DataSet transitions_dataset = file.openDataSet("sentence1_transitions");
    char *transitions_data[1];
    read_variable_length_data(file, transitions_dataset, transitions_data, offset);

    string transitions_string = transitions_data[0];
    auto transitions_json = json::parse(transitions_string);
    cout << transitions_json << endl;
    int transitions_sample = transitions_json[0];
    cout << transitions_sample << endl;

    free(transitions_data[0]);
    transitions_dataset.close();
}

map<string, int> tokenize_example(H5File &file)
{
    /*

    $ ./demo -input_file dev_matched.h5
    2017-06-09 16:40:49 DEBUG [tokenize_example] [main.cpp:102] Tokenizing.
    2017-06-09 16:40:50 DEBUG [tokenize_example] [main.cpp:130] Done tokenizing.

    $ ./demo -input_file train.h5
    2017-06-09 16:40:52 DEBUG [tokenize_example] [main.cpp:102] Tokenizing.
    2017-06-09 16:41:27 DEBUG [tokenize_example] [main.cpp:130] Done tokenizing.

    */

    map<string, int> token_to_index;
    int next_index = 0;

    LOGDEBUG("Tokenizing.");

    // Tokens example.
    DataSet tokens_dataset = file.openDataSet("sentence1_tokens");
    int num_tokens = tokens_dataset.getSpace().getSelectNpoints();
    char *tokens_data[1];
    int offset;
    for (int i = 0; i < num_tokens; i++)
    {
        offset = i;
        read_variable_length_data(file, tokens_dataset, tokens_data, offset);
        string tokens_string = tokens_data[0];
        auto tokens_json = json::parse(tokens_string);
        for (int j = 0; j < tokens_json.size(); j++)
        {
            string tokens_sample = tokens_json[j];
            map<string, int>::iterator it = token_to_index.find(tokens_sample);
            if (it != token_to_index.end())
            {
                // pass
            } else {
                token_to_index.insert(pair<string,int>(tokens_sample, next_index));
                next_index++;
            }
        }
        free(tokens_data[0]);
    }

    LOGDEBUG("Done tokenizing.");

    tokens_dataset.close();

    return token_to_index;
}

int main(int argc, char *argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    gflags::ShutDownCommandLineFlags();

    H5File file(FLAGS_input_file, H5F_ACC_RDONLY);

    // dataset_example(file);
    // tokens_example(file);
    // transitions_example(file);
    tokenize_example(file);

    file.close();

    return 0;
}
