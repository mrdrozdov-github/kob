#include <math.h>
#include <map>

#include <iostream>
#include <fstream>
#include <sstream>

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

- [x] Properly tokenize sentences.
- [x] Read batches (tokens and transitions)
- [x] Pad sentences (and transitions)
- [ ] Full example.

*/

DEFINE_string(input_file, "dev_matched.h5", "Data file");

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

struct NLIObject
{
    vector<string> sentence1_tokens;
    vector<string> sentence2_tokens;
    vector<int> sentence1_transitions;
    vector<int> sentence2_transitions;
    int label;

    NLIObject() {}
};

template<typename T>
void read_dataset(H5File &file, string datasetname, vector<T> &out_vec, int offset)
{
    char *out[1];
    DataSet dataset = file.openDataSet(datasetname);
    read_variable_length_data(file, dataset, out, offset);
    string out_str = out[0];
    auto out_json = json::parse(out_str);
    for (int i = 0; i < out_json.size(); i++)
    {
        out_vec.push_back(out_json[i]);
    }
    free(out[0]);
    dataset.close();
}

void read_dataset(H5File &file, string datasetname, void *out, int offset)
{
    DataSet dataset = file.openDataSet(datasetname);
    read_variable_length_data(file, dataset, out, offset);
    dataset.close();
}

NLIObject *read_example(H5File &file, int offset)
{
    NLIObject *result = new NLIObject;

    read_dataset<string>(file, "sentence1_tokens", result->sentence1_tokens, offset);
    read_dataset<int>(file, "sentence1_transitions", result->sentence1_transitions, offset);
    read_dataset<string>(file, "sentence2_tokens", result->sentence2_tokens, offset);
    read_dataset<int>(file, "sentence2_transitions", result->sentence2_transitions, offset);
    read_dataset(file, "labels", &(result->label), offset);

    return result;
}

// source: https://stackoverflow.com/a/236803/1185578
template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

// source: https://stackoverflow.com/a/236803/1185578
std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

pair<THFloatTensor *, map<string, int>> read_embeddings(string filename, map<string, int> token_to_index, int embedding_size)
{
    /*

    2017-06-10 12:38:05 DEBUG [read_embeddings] [main.cpp:237] Reading: /Users/Andrew/data/glove.6B.100d.txt
    2017-06-10 12:38:38 DEBUG [read_embeddings] [main.cpp:272] Done.
    2017-06-10 12:38:38 DEBUG [read_embeddings] [main.cpp:278] Found 16 in vocab.

    */

    map<string, int> new_token_to_index; // reassign indices based on the order they appear in embedding file
    int next_index = 0;

    // 1. Allocate an embedding tensor with size of original vocab
    int vocab_size = token_to_index.size();
    THFloatTensor *embeddings = THFloatTensor_newWithSize2d(vocab_size, embedding_size);

    // 2. Read embedding file line by line. If in vocab, create a new item in the new token_to_index and
    //      and add a row to the embedding tensor.

    LOGDEBUG("Reading: %s", filename.c_str());

    // 
    ifstream file(filename);
    string linebuffer;

    while (file && getline(file, linebuffer)){
        if (linebuffer.length() == 0) continue;

        // Split line.
        vector<string> elems = split(linebuffer, ' ');
        string key = elems[0];

        // Check if token exists in vocab.
        int ii;
        map<string, int>::iterator it = token_to_index.find(key);
        if (it != token_to_index.end())
        {
            // Read single embedding.
            ii = 0;
            for (vector<string>::iterator cur = elems.begin() + 1; cur != elems.end(); ++cur) {
                float val = stof(*cur);
                THFloatTensor_set2d(embeddings, next_index, ii, val);
                ii++;
            }

            // Update new token_to_index.
            new_token_to_index.insert(pair<string,int>(key, next_index));
            next_index++;
        }

        // TODO: Remove this.
        if (next_index > 10) {
            break;
        }

    }
    LOGDEBUG("Done.");

    // 3. When complete, slice the embedding tensor, keeping only the rows that have been assigned.
    THFloatTensor *out_embeddings = THFloatTensor_newWithSize2d(vocab_size, next_index);
    THFloatTensor_narrow(out_embeddings, embeddings, 0, 0, next_index);
    LOGDEBUG("Found %d in vocab.", next_index);

    return pair<THFloatTensor *, map<string, int>>(out_embeddings, new_token_to_index);
}

void simple_embed(THFloatTensor *out, vector<string> tokens, map<string, int> token_to_index, THFloatTensor *embeddings)
{
    int embedding_size = embeddings->size[1];
    THFloatTensor *_out = THFloatTensor_new();
    THFloatTensor *_embedding = THFloatTensor_new();
    for (int i = 0; i < tokens.size(); i++)
    {
        string token = tokens[i];
        map<string, int>::iterator it = token_to_index.find(token);
        if (it != token_to_index.end())
        {
            int index = (*it).second;

            THFloatTensor_select(_out, out, 0, i);
            THFloatTensor_select(_embedding, embeddings, 0, index);
            THFloatTensor_copy(_out, _embedding);
        } else {
            // do nothing (or fill with zeros. this is an UNK!)
        }
    }
}

void embed_example(H5File &file)
{
    int batch_size = 3;
    int seq_length = 100;
    int embedding_size = 100;
    NLIObject *result[batch_size];

    for (int b = 0; b < batch_size; b++)
    {
        result[b] = read_example(file, b);
    }

    vector<string> tokens;
    map<string, int> token_to_index; // TODO: This can simply be a set.
    int next_index = 0;

    for (int b = 0; b < batch_size; b++)
    {
        // Tokens for sentence1.
        tokens = result[b]->sentence1_tokens;
        for (int i = 0; i < tokens.size(); i++)
        {
            string sample = tokens[i];
            map<string, int>::iterator it = token_to_index.find(sample);
            if (it != token_to_index.end())
            {
                // pass
            } else {
                token_to_index.insert(pair<string,int>(sample, next_index));
                next_index++;
            }
        }

        // Tokens for sentence2.
        tokens = result[b]->sentence2_tokens;
        for (int i = 0; i < tokens.size(); i++)
        {
            string sample = tokens[i];
            map<string, int>::iterator it = token_to_index.find(sample);
            if (it != token_to_index.end())
            {
                // pass
            } else {
                token_to_index.insert(pair<string,int>(sample, next_index));
                next_index++;
            }
        }
    }

    pair<THFloatTensor *, map<string, int>> embeddings_out =
        read_embeddings("/Users/Andrew/data/glove.6B.100d.txt", token_to_index, embedding_size);

    THFloatTensor *embeddings = embeddings_out.first;
    map<string, int> embed_token_to_index = embeddings_out.second;

    // Example batch.
    // TODO: Use lookup table.
    THFloatTensor *batch = THFloatTensor_newWithSize3d(batch_size*2, seq_length, embedding_size);
    THFloatTensor_fill(batch, 0.0);
    THFloatTensor *batch_row = THFloatTensor_new();
    for (int b = 0; b < batch_size; b++)
    {
        tokens = result[b]->sentence1_tokens;
        THFloatTensor_select(batch_row, batch, 0, b);
        simple_embed(batch_row, tokens, embed_token_to_index, embeddings);
    }
}

int main(int argc, char *argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    gflags::ShutDownCommandLineFlags();

    H5File file(FLAGS_input_file, H5F_ACC_RDONLY);

    // dataset_example(file);
    // tokens_example(file);
    // transitions_example(file);
    // tokenize_example(file);
    embed_example(file);

    file.close();

    return 0;
}
