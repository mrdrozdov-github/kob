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

// Progressbar.
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

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
DEFINE_int32(batch_size, 32, "Batch size");
DEFINE_string(embedding_file, "/Users/Andrew/data/glove.6B.100d.txt", "GloVe embedding file");
DEFINE_int32(embedding_size, 100, "Embedding size");
DEFINE_int32(seq_length, 300, "Sequence length");
DEFINE_int32(hidden_dim, 100, "Hidden dimension");
DEFINE_int32(num_classes, 3, "Number of classes");

void read_variable_length_data(H5File &file, DataSet &dataset, void *out, int _offset)
{
    /* Get datatype for dataset */
    DataType dtype = dataset.getDataType();

    hsize_t offset[1], count[1], stride[1], block[1];
    hsize_t dims[1], dimsm[1];

    int num_examples = dataset.getSpace().getSelectNpoints();
    dims[0] = num_examples;
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

void printProgress(double percentage)
{
    // https://stackoverflow.com/a/36315819/1185578
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf ("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush (stdout);
}

void finishProgress()
{
    printf("\n");
}

struct NLIObject
{
    vector<string> sentence1_tokens;
    vector<string> sentence2_tokens;
    vector<int> sentence1_transitions;
    vector<int> sentence2_transitions;
    int label;

    NLIObject() {}
    ~NLIObject() {}
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

        // TODO Remove
        if (ii > 10) {
            break;
        }

    }

    // 3. When complete, slice the embedding tensor, keeping only the rows that have been assigned.
    THFloatTensor *out_embeddings = THFloatTensor_newWithSize2d(vocab_size, next_index);
    THFloatTensor_narrow(out_embeddings, embeddings, 0, 0, next_index);

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

int get_num_examples(H5File &file)
{
    DataSet dataset = file.openDataSet("labels");
    int num_examples = dataset.getSpace().getSelectNpoints();
    return num_examples;
}

map<string, int> get_initial_tokens(H5File &file)
{
    int batch_size = FLAGS_batch_size;
    int seq_length = FLAGS_seq_length;
    int num_examples = get_num_examples(file);
    NLIObject *result;

    vector<string> tokens;
    map<string, int> token_to_index; // TODO: This can simply be a set.
    int next_index = 0;

    for (int i_example = 0; i_example < num_examples; i_example++)
    {
        result = read_example(file, i_example);

        // Tokens for sentence1.
        tokens = result->sentence1_tokens;
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
        tokens = result->sentence2_tokens;
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

        free(result);
    }

    return token_to_index;
}

class MLP
{
    private:
        int input_dim;
        int hidden_dim;
        int output_dim;
    public:
        MLP(int input_dim, int hidden_dim, int output_dim);
        ~MLP();
        Variable *forward(Variable *x);
        void clear_grads();
        Variable *inputs[3];
        Variable *outputs[3];
        Linear *layers[2];
};

MLP::MLP(int input_dim, int hidden_dim, int output_dim) {
    this->input_dim = input_dim;
    this->hidden_dim = hidden_dim;
    this->output_dim = output_dim;

    this->layers[0] = new Linear(input_dim, hidden_dim);
    this->layers[1] = new Linear(hidden_dim, output_dim);
}

MLP::~MLP() {
    delete this->layers[0];
    delete this->layers[1];
}

void MLP::clear_grads() {
    this->layers[0]->clear_grads();
    this->layers[1]->clear_grads();
}

Variable *MLP::forward(Variable *x) {
    // Linear
    this->inputs[0] = x;
    this->outputs[0] = this->layers[0]->call(this->inputs[0]);

    // Sigmoid
    this->inputs[1] = this->outputs[0];
    this->outputs[1] = F_sigmoid(this->inputs[1]);

    // Linear
    this->inputs[2] = this->outputs[1];
    this->outputs[2] = this->layers[1]->call(this->inputs[1]);

    return this->outputs[2];
}

void run(H5File &file)
{
    int batch_size = FLAGS_batch_size;
    int seq_length = FLAGS_seq_length;
    int embedding_size = FLAGS_embedding_size;
    int max_epochs = 1;
    int num_examples = get_num_examples(file);
    int num_batches = num_examples / batch_size;

    MLP mlp = MLP(FLAGS_embedding_size * 2, FLAGS_hidden_dim, FLAGS_num_classes);


    // 1.
    LOGDEBUG("Reading tokens: start.");
    map<string, int> token_to_index = get_initial_tokens(file);
    LOGDEBUG("Found %lu tokens.", token_to_index.size());
    LOGDEBUG("Reading tokens: done.");


    // 2.
    LOGDEBUG("Reading embeddings: start.");
    LOGDEBUG("Reading: %s", FLAGS_embedding_file.c_str());
    pair<THFloatTensor *, map<string, int>> embeddings_out =
        read_embeddings(FLAGS_embedding_file, token_to_index, embedding_size);

    THFloatTensor *embeddings = embeddings_out.first;
    map<string, int> embed_token_to_index = embeddings_out.second;
    LOGDEBUG("Found %lu tokens.", embed_token_to_index.size());
    LOGDEBUG("Reading embeddings: done.");


    // 3.
    LOGDEBUG("Iterating over dataset: start.");
    for (int epoch = 0; epoch < max_epochs; epoch++) {
        for (int i_batch = 0; i_batch < num_batches; i_batch++) {
            NLIObject *batch_objects[batch_size];
            vector<string> tokens;

            // Read batches.
            for (int b = 0; b < batch_size; b++) {
                batch_objects[b] = read_example(file, b); // TODO: Use shuffled indices.
            }

            // Create batched & embedded tokens.
            THFloatTensor *batch = THFloatTensor_newWithSize3d(batch_size * 2, seq_length, embedding_size);
            THFloatTensor_fill(batch, 0.0);
            THFloatTensor *batch_row = THFloatTensor_new();

            THLongTensor *target = THLongTensor_newWithSize1d(batch_size);

            // Embed sentence1.
            for (int b = 0; b < batch_size; b++) {
                tokens = batch_objects[b]->sentence1_tokens;
                THFloatTensor_select(batch_row, batch, 0, b);
                simple_embed(batch_row, tokens, embed_token_to_index, embeddings);
            }

            // Embed sentence2.
            for (int b = 0; b < batch_size; b++) {
                tokens = batch_objects[b]->sentence2_tokens;
                THFloatTensor_select(batch_row, batch, 0, batch_size + b);
                simple_embed(batch_row, tokens, embed_token_to_index, embeddings);
            }

            // Labels.
            for (int b = 0; b < batch_size; b++) {
                THLongTensor_set1d(target, b, batch_objects[b]->label);
            }

            // Fix labels.
            THLongTensor_add(target, target, 1);

            // Cleanup.
            for (int b = 0; b < batch_size; b++) {
                free(batch_objects[b]);
            }

            // Prepare input.
            
            // 1. Sum
            THFloatTensor *summed = THFloatTensor_new();
            THFloatTensor_sum(summed, batch, 1, true);
            THFloatTensor_resize2d(summed, batch_size * 2, embedding_size);

            // 2. Narrow
            THFloatTensor *sent1 = THFloatTensor_new();
            THFloatTensor *sent2 = THFloatTensor_new();
            THFloatTensor_narrow(sent1, summed, 0, 0, batch_size);
            THFloatTensor_narrow(sent2, summed, 0, batch_size, batch_size);

            // 3. Concat
            THFloatTensor *h = THFloatTensor_new();
            THFloatTensor_cat(h, sent1, sent2, 1);

            // 4. Forward pass.
            Variable *logits = mlp.forward(new Variable(h));

            // 4a. Loss.
            Variable *probs = F_log_softmax(logits);
            Variable *loss = F_nll(probs, target);

            // 5. Backward pass.
            mlp.clear_grads();


            delete loss;
            delete probs;
            delete logits;
            for (int ii = 0; ii < 3; ii++) {
                delete mlp.inputs[ii];
            }


            // Cleanup.
            THFloatTensor_free(h);
            THFloatTensor_free(sent1);
            THFloatTensor_free(sent2);
            THFloatTensor_free(summed);
            THFloatTensor_free(batch_row);
            THFloatTensor_free(batch);

            printProgress((i_batch + 1) / (float)num_batches);
        }
        finishProgress();
        LOGDEBUG("Finished epoch: %d", epoch + 1);
    }
    LOGDEBUG("Iterating over dataset: done.");
}

int main(int argc, char *argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    gflags::ShutDownCommandLineFlags();

    H5File file(FLAGS_input_file, H5F_ACC_RDONLY);

    run(file);

    file.close();

    return 0;
}
