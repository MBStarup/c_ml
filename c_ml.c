#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>

#define DATA_SIZE 784
#define DATA_WIDTH 28
#define DATA_HEIGHT 28
#define DATA_AMOUNT 12
#define PRINT_NUM 5
#define EPOCHS 3000
#define BATCH_SIZE 4
#define PRINTED_EXAMPLE 0
#define PRINTED_EXAMPLE_AMOUNT 12
#define SHUFFLE_N 100
#define _NO_PRINT

#define DEBUG(fmt, var) printf("%s: " fmt, #var, var)
#define SET_RED() printf("\e[31m")
#define SET_YELLOW() printf("\e[93m")
#define SET_GREEN() printf("\e[92m")
#define SET_RESET() printf("\e[0m")
#define DEBUG_LF(var)            \
    do                           \
    {                            \
        if (var < 0)             \
            SET_RED();           \
        if (var == 0)            \
            SET_YELLOW();        \
        if (var > 0)             \
            SET_GREEN();         \
        DEBUG("%+012.5lf", var); \
        SET_RESET();             \
    } while (0)

typedef struct
{
    int label;
    double img[DATA_SIZE];
    double expected[10];
} image;

typedef struct
{
    size_t in;
    size_t out;
    double *weights;
    double *biases;
    // double (*func)(double);
} layer;

typedef struct
{
    size_t layer_capacity;
    size_t layer_amount;
    layer *layers;
    char *name;
} model;

// Allocates 'size' bytes initialized to 0 and asserts that the allocation succeeded
// Memory is still freed with free()
void *ass_calloc(size_t size)
{
    void *ptr = calloc(size, 1);
    assert(ptr != NULL);
    return ptr;
}

// Allocates 'size' bytes and asserts that the allocation succeeded
// Memory is still freed with free()
void *ass_malloc(size_t size)
{
    void *ptr = malloc(size);
    assert(ptr != NULL);
    return ptr;
}

// Shuffles an array by repeadedly picking two random indexes and swapping them arr_length * SHUFFLE_N times
// ------------------------------
// arr_length: the amount of elements in the array, accepted values: {1 .. SIZE_MAX}
// elem_size: the size of each element in bytes, accepted values: {1 .. SIZE_MAX}
// arr: the array to shuffle
void shuffle_arr(size_t arr_length, size_t elem_size, void *arr)
{
    typedef unsigned char byte;
    assert(sizeof(byte) == 1);

    assert(arr_length > 0); // Cannot shuffle arrays of length zero
    assert(elem_size > 0);  // Cannot shuffle arrays with zero size elements
    assert(arr != NULL);    // Cannot shuffle NULL

    byte *array = (byte *)arr;

    byte *temp = ass_malloc(elem_size); // A temp variable to store a value while we shuffle

    for (size_t i = 0; i < arr_length * SHUFFLE_N; i++)
    {
        // pick two random indicies in the arr
        size_t a = (size_t)(((double)rand() / (double)RAND_MAX) * (arr_length)); // Shouldn't this be "... * (arr_length - 1)"? Although when I do that it seems to never shuffle the last one so...
        size_t b = (size_t)(((double)rand() / (double)RAND_MAX) * (arr_length)); // Shouldn't this be "... * (arr_length - 1)"? Although when I do that it seems to never shuffle the last one so...
        // if (a == PRINTED_EXAMPLE || b == PRINTED_EXAMPLE)
        //     printf("Shufflin %d and %d\n", a, b);
        memcpy(temp, array + (a * elem_size), elem_size);                    // temp = arr[a]
        memcpy(array + (a * elem_size), array + (b * elem_size), elem_size); // arr[a] = arr[b]
        memcpy(array + (b * elem_size), temp, elem_size);                    // arr[b] = temp
    }

    free(temp);
}

void __shuffle_test(size_t count)
{
    int arr[count];
    for (size_t i = 0; i < count; i++)
    {
        arr[i] = i;
    }

    printf("\nBefore: \n");
    for (size_t i = 0; i < count; i++)
    {
        printf("%3d ", arr[i]);
        if (i % 10 == 9)
            printf("\n");
    }

    shuffle_arr(count, sizeof(int), arr);
    printf("\nAfter: \n");
    for (size_t i = 0; i < count; i++)
    {
        printf("%3d ", arr[i]);
        if (i % 10 == 9)
            printf("\n");
    }

    printf("\nMatches: \n");
    for (size_t i = 0; i < count; i++)
    {
        if (arr[i] == i)
            printf("%3d ", arr[i]);
        else
            printf("    ");

        if (i % 10 == 9)
            printf("\n");
    }

    return;
}

void randomize_double_arr(double *arr, int size, double min, double max)
{
    for (size_t i = 0; i < size; i++)
    {
        arr[i] = min + (((double)rand()) / ((double)RAND_MAX)) * (max - min);
    }
}

void softmax(int size, double *inputs, double *outputs)
{
    double *e_arr = ass_malloc(sizeof(double) * size);
    double accum = 0;

    for (size_t i = 0; i < size; i++)
    {
        e_arr[i] = exp(inputs[i]);
        accum += e_arr[i];
    }

    for (size_t i = 0; i < size; i++)
    {
        outputs[i] = e_arr[i] / accum;
    }
    free(e_arr);
}

double sum(int size, double *inputs)
{
    double accum = 0;

    for (size_t i = 0; i < size; i++)
    {
        accum += inputs[i];
    }
    return accum;
}

layer layer_new(int in, int out /*, double (*func)(double) */)
{
    layer res;

    res.in = in;
    res.out = out;

    res.weights = ass_malloc(sizeof(double) * in * out);
    randomize_double_arr(res.weights, in * out, 0, 1);

    res.biases = ass_malloc(sizeof(double) * out);
    randomize_double_arr(res.biases, out, 0, 1);

    // res.func = func;

    return res;
}

void layer_del(layer l)
{
    free(l.biases);
    free(l.weights);
}

// Calculates the weigted sum, does not apply any activation function
// ----------------------------
// Assumes the size of inputs matches the size of l.in
// if greater, will cut off remaning inputs
// if smaller will potentially segfault
// Assumes the size of outputs matches the size of l.out
// writes results to outputs
void layer_apply(layer l, double *inputs, double *outputs)
{
    for (size_t i_out = 0; i_out < l.out; i_out++)
    {
        double accum = 0;
        for (size_t i_in = 0; i_in < l.in; i_in++)
        {
            accum += l.weights[i_out * l.in + i_in] * (inputs[i_in]);
        }
        outputs[i_out] = accum + l.biases[i_out];
    }
}

model model_new(char *name, size_t capacity)
{
    model result;
    result.layer_capacity = capacity;
    result.layer_amount = 0;
    result.name = name;
    result.layers = ass_malloc(sizeof(layer) * capacity);
    return result;
}

void model_del(model model)
{
    for (size_t layer = 0; layer < model.layer_amount; layer++)
    {
        layer_del(model.layers[layer]);
    }

    free(model.layers);
    model.layer_amount = 0;
    model.layer_capacity = 0;
}

void model_add(model *model, layer layer)
{
#ifndef NO_PRINT
    printf("adding layer %d of %d to model %s\n", model->layer_amount + 1, model->layer_capacity, model->name);
#endif
    assert(model->layer_amount < model->layer_capacity);
    if (model->layer_amount > 0)
    {
        assert(model->layers[model->layer_amount - 1].out == layer.in);
    }
    model->layers[model->layer_amount] = layer;
    model->layer_amount += 1;
}

image parse_line(char *line)
{
    image result;
    char *token = strtok(line, ",");
    result.label = atoi(token);
    for (size_t i = 0; i < 10; i++)
    {
        result.expected[i] = 0;
    }

    result.expected[result.label] = 1;

    for (int i = 0; i < DATA_SIZE; i++)
    {
        token = strtok(NULL, ",");
        result.img[i] = ((double)atoi(token)) / 255;
    }
    return result;
}

inline void print_image_data(image d)
{
    SET_GREEN();
    printf("Label: %d\n", d.label);
    for (int i = 0; i < DATA_WIDTH; i++)
    {
        for (int j = 0; j < DATA_HEIGHT; j++)
        {
            d.img[i * DATA_WIDTH + j] > 0 ? printf("  ") : printf("[]");
        }
        printf("\n");
    }
    SET_RESET();
}

inline void print_double_arr(size_t size, double *arr)
{
    for (size_t i = 0; i < size; i++)
    {
        printf("%+012.5lf, ", arr[i]);
        if (i % PRINT_NUM == (PRINT_NUM - 1) && i + 1 < size)
        {
            printf("\n");
        }
    }
}

// return (x > 0) ? x : 0;
inline double relu(double x)
{
    return (x > 0) ? x : 0;
}

// return x > 0;
inline double derivative_of_relu(double x)
{
    return x > 0;
}

// return 1.0 / (1.0 + exp(-1 * x));
inline double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-1 * x));
}

// return sigmoid(x) * (1 - sigmoid(x));
inline double derivative_of_sigmoid(double x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

inline double x(double x)
{
    return x;
}

double cost(int size, double *actual, double *expected)
{
    double result = 0;
    for (int i = 0; i < size; i++)
    {
        double delta = actual[i] - expected[i];
        result += delta * delta;
    }
    return result;
}

int main(int argc, char const *argv[])
{
    image *data = ass_malloc(sizeof(image) * DATA_AMOUNT);

    // file stuff
    {
        printf("START\n");
        const char *TRAIN_DATA_PATH = "mnist_train.csv";
        const char *TEST_DATA_PATH = "mnist_test.csv";
        printf("Opening file: %s\n", TRAIN_DATA_PATH);
        FILE *fptr = fopen(TRAIN_DATA_PATH, "r");

        const int file_buffer_size = 4 * DATA_SIZE + 2;
        char file_buffer[file_buffer_size];

        fgets(file_buffer, file_buffer_size, fptr);
        fgets(file_buffer, file_buffer_size, fptr);

        printf("\n");
        for (size_t i = 0; i < DATA_AMOUNT; i++)
        {
            char *line = fgets(file_buffer, file_buffer_size, fptr);
            assert(line != NULL);
            data[i] = parse_line(line);
#ifndef NO_PRINT
            printf("\x1b[1F");
            printf("Parsing data: %d/%d\n", i + 1, DATA_AMOUNT);
#endif
        }
        fclose(fptr);
    }
    srand(420);
    // srand(time(0));
    model model = model_new("test_model", 5);
    {

        /*
         *         Input[t]    WeightIH   Hidden[t]  WeightHO  Output[t]
         *          input      --(IH)->      H       --(HO)->     O
         *
         *    results[t*(bs) + 0]  model.layers[0]  results[t*(bs) + 1]  model.layers[1]  results[t*(bs) + 2]  model.layers[2]  results[t*(bs) + 3]  model.layers[3]  results[t*(bs) + 4]
         *          input            --(I_H0)->            H0              --(H0_H1)->          H1               --(H1_H2)->           H2              --(H2_O)->             O
         *
         *
         *                   Target - Output => results[t*(bs) + 2] - results[t*(bs) + 1]
         *
         */

        model_add(&model, layer_new(DATA_SIZE, 8));
        model_add(&model, layer_new(8, 10));
    }

    for (size_t i = 1; i < model.layer_amount; i++)
    {
        printf("\n--------------------\n%d:\n", i);
        print_double_arr(model.layers[i].in * model.layers[i].out, model.layers[i].weights);
    }

    printf("Starting Training\n\n");
    double one_over_batch_size = 1.0 / (double)BATCH_SIZE;
    size_t batch_amount = DATA_AMOUNT / BATCH_SIZE;
    assert(batch_amount * BATCH_SIZE == DATA_AMOUNT); // DATA_AMOUNT shoudl be divisble by BATCH_SIZE
    double total_cost = 0;
    for (size_t epoch = 0; epoch < EPOCHS; epoch++)
    {
        // Status printing
        if (epoch % 100 == 0)
        {
            DEBUG("%d: ", epoch);
            DEBUG_LF(total_cost);
            printf("\n");
            if (epoch % 10000 == 0)
            {
                for (size_t i = 1; i < model.layer_amount; i++)
                {
                    printf("\n--------------------\n%d:\n", i);
                    print_double_arr(model.layers[i].in * model.layers[i].out, model.layers[i].weights);
                }
            }
        }
        total_cost = 0;

        // Shuffle data order
        shuffle_arr(DATA_AMOUNT, sizeof(data[0]), data);

        // for every batch of training
        for (size_t batch = 0; batch < batch_amount; batch++)
        {
            // for every training example in the given batch, allocate a buffer for the input data, followed by a buffer for the intermediate results between the layers
            // then fill these buffers by using the neural net
            double **results = ass_malloc(sizeof(double *) * (BATCH_SIZE * (model.layer_amount + 1)));
            for (int i = 0; i < BATCH_SIZE; i++)
            {
                results[i * (model.layer_amount + 1) + 0] = data[batch * BATCH_SIZE + i].img; // set the zero'th "result" to the input data
                for (size_t layer = 0; layer < model.layer_amount; layer++)
                {
                    results[i * (model.layer_amount + 1) + (layer + 1)] = ass_malloc(sizeof(double) * model.layers[layer].out); // TODO: free
                }

                double *activated_results = ass_malloc(sizeof(double) * model.layers[0].out);

                layer_apply(model.layers[0], data[batch * BATCH_SIZE + i].img, results[i * (model.layer_amount + 1) + (0 + 1)]); // layer 0, plus an offset of 1 for the input data at the front
                for (size_t layer = 1; layer < model.layer_amount; layer++)
                {
                    // apply activation function to the outputs from the prevoious layer
                    for (size_t neuron = 0; neuron < model.layers[layer].in; neuron++)
                    {
                        activated_results[neuron] = relu(results[i * (model.layer_amount + 1) + (layer + 1) - 1][neuron]);
                    }

                    // Apply the layer, use the activated outputs from the prev layer as the input
                    layer_apply(model.layers[layer], activated_results, results[i * (model.layer_amount + 1) + (layer + 1)]);
                    free(activated_results);
                    activated_results = ass_malloc(sizeof(double) * model.layers[layer].out);
                }

                for (size_t neuron = 0; neuron < model.layers[model.layer_amount - 1].out; neuron++)
                {
                    activated_results[neuron] = relu(results[i * (model.layer_amount + 1) + ((model.layer_amount - 1) + 1) - 1][neuron]);
                }

                total_cost += cost(model.layers[model.layer_amount - 1].out, activated_results, data[batch * BATCH_SIZE + i].expected);

                free(activated_results);
            }

            // for every training in the batch, backpropegate and calculate the weight adjustments
            for (size_t training = 0; training < BATCH_SIZE; training++)
            {

                // For the final layer, the dcost_dout for a given output neuron is given by the derivative of the cost function with respect to that neuron (yes?) (maybe multiply by 2?)
                double *dcost_dout = ass_malloc(sizeof(double) * model.layers[model.layer_amount - 1].out);
                for (size_t output_neuron = 0; output_neuron < model.layers[model.layer_amount - 1].out; output_neuron++)
                {
                    dcost_dout[output_neuron] = data[batch * BATCH_SIZE + training].expected[output_neuron] - relu(results[(training) * (model.layer_amount + 1) + (model.layer_amount - 1 + 1)][output_neuron]);
                }

                for (int layer = model.layer_amount - 1; layer >= 0; layer--)
                {
                    double *next_dcost_dout = ass_calloc(sizeof(double) * model.layers[layer].in);
                    double one_over_input_amount = (double)1.0 / model.layers[layer].in;
                    for (size_t output_neuron = 0; output_neuron < model.layers[layer].out; output_neuron++)
                    {
                        double step_size = 0.000000001;
                        double dout_dz = derivative_of_relu(results[(training) * (model.layer_amount + 1) + (layer + 1)][output_neuron]); // 1 if the result of a given output is greater than 0, else 0

                        for (size_t input_neuron = 0; input_neuron < model.layers[layer].in; input_neuron++)
                        {
                            // dz_dw is just the input neuron (after activation)
                            double dz_dw = relu(results[(training) * (model.layer_amount + 1) + (layer + 1) - 1][input_neuron]);

                            // The adjustment to the weight is proportional to dCost_dWeight, ie. how big an influence the weight has on the cost function
                            // summed up over each training in the batch
                            model.layers[layer].weights[output_neuron * model.layers[layer].in + input_neuron] += step_size * dcost_dout[output_neuron] * dout_dz * dz_dw;
                            printf("\n");
                            DEBUG("%d ", output_neuron);
                            DEBUG_LF(dcost_dout[output_neuron]);
                            printf("\n");
                            exit(0);

                            // dCost_dOut for a given output neuron in the previous layer (ie. a given input in our current layer), is proportional to the weight between that neuron and the current neuron, and the current neurons dCost_dz
                            // summed up over all the neurons that connect to it
                            next_dcost_dout[input_neuron] += model.layers[layer].weights[output_neuron * model.layers[layer].in + input_neuron] * dcost_dout[output_neuron] * dout_dz;
                        }
                        // The adjustment to the bias is proportional to that neurons influence on the cost function
                        model.layers[layer].biases[output_neuron] += step_size * dcost_dout[output_neuron] * dout_dz;
                    }
                    free(dcost_dout);
                    dcost_dout = next_dcost_dout;
                }
                free(dcost_dout);
            }
        }
    }

    // use the trained model
    for (size_t i = 0; i < PRINTED_EXAMPLE_AMOUNT; i++)
    {
        {
            printf("\n_____________________________\n");
            printf("\nExample image nr. %d:\n", PRINTED_EXAMPLE + i);
            print_image_data(data[PRINTED_EXAMPLE + i]); // print the example image
            printf("\n");
            double **final_results = ass_malloc(sizeof(double *) * model.layer_amount);

            for (size_t layer = 0; layer < model.layer_amount; layer++)
            {
                final_results[layer] = ass_malloc(sizeof(double) * model.layers[layer].out); // TODO: free
            }

            // apply first layer manually, since there's no previous result to use as input
            layer_apply(model.layers[0], data[PRINTED_EXAMPLE + i].img, final_results[0]);
            for (size_t output = 0; output < model.layers[0].out; output++)
            {
                final_results[0][output] = relu(final_results[0][output]); // apply activation function
            }
            // print for debug
            printf("Layer 0 results: \n");
            print_double_arr(model.layers[0].out, final_results[0]);
            printf("\n------------------\n");

            // for the rest of the layers apply them in a loop
            for (size_t layer = 1; layer < model.layer_amount; layer++)
            {
                layer_apply(model.layers[layer], final_results[layer - 1], final_results[layer]); // Apply the layer, use the output from the prev layer as the input
                for (size_t output = 0; output < model.layers[layer].out; output++)
                {
                    final_results[layer][output] = relu(final_results[layer][output]); // apply activation function
                }

                // print for debug
                printf("Layer %d results :\n", layer + 1);
                print_double_arr(model.layers[layer].out, final_results[layer]);
                printf("\n------------------\n");
            }

            int out = 1;
            int layer = model.layer_amount - 1;
            printf("Output %d in layer %d with example %d = %+012.5lf | calulated as such:\n", out, layer, PRINTED_EXAMPLE + i, final_results[layer][out]);
            double acc = model.layers[layer].biases[out];
            printf("%+012.5lf\n", acc);
            for (size_t in = 0; in < model.layers[layer].in; in++)
            {
                double input = (layer == 0 ? data[PRINTED_EXAMPLE + i].img[in] : relu(final_results[layer - 1][in]));
                double w = model.layers[layer].weights[out * model.layers[layer].in + in];
                acc += w * input;
                printf(" + (");
                DEBUG_LF(w);
                SET_RESET();
                printf(" * ");
                DEBUG_LF(input);
                printf(" ) = ");
                DEBUG_LF(acc);
                printf("\n");
                SET_RESET();
            }
            printf("\n");

            // apply softmax on the last result to get probability result
            // softmax(model.layers[model.layer_amount - 1].out, final_results[model.layer_amount - 1], final_results[model.layer_amount - 1]);

            free(final_results);
        }
    }

    model_del(model);
    free(data);
    printf("END\n");
    return 0;
}