#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>

// -- ml config --
#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define TRAINING_DATA_AMOUNT 12
#define EPOCHS 3000
#define BATCH_SIZE 4

// -- other config --
#define DATA_WIDTH 28
#define DATA_HEIGHT 28
#define PRINT_NUM 5
#define PRINTED_EXAMPLE 0
#define PRINTED_EXAMPLE_AMOUNT 12
#define SHUFFLE_N 100
#define _NO_PRINT

// -- debugging tools --
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
    double img[INPUT_SIZE];
    double expected[OUTPUT_SIZE];
} image;

typedef struct
{
    size_t in;
    size_t out;
    double *weights;
    double *biases;
    // double (*func)(double);
} layer;

int alloc_counter = 0;

// Allocates 'size' bytes initialized to 0 and asserts that the allocation succeeded
// Memory is still freed with ass_free()
void *ass_calloc(size_t size)
{
    ++alloc_counter;
    void *ptr = calloc(size, 1);
    assert(ptr != NULL);
    return ptr;
}

// Allocates 'size' bytes and asserts that the allocation succeeded
// Memory is still freed with ass_free()
void *ass_malloc(size_t size)
{
    ++alloc_counter;
    void *ptr = malloc(size);
    assert(ptr != NULL);
    return ptr;
}

void ass_free(void *ptr)
{
    --alloc_counter;
    free(ptr);
}

void randomize_double_arr(double *arr, int size, double min, double max)
{
    for (int i = 0; i < size; i++)
    {
        arr[i] = min + (((double)rand()) / ((double)RAND_MAX)) * (max - min);
    }
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

    for (int i = 0; i < arr_length * SHUFFLE_N; i++)
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

    ass_free(temp);
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
    ass_free(l.biases);
    ass_free(l.weights);
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
    for (int i_out = 0; i_out < l.out; i_out++)
    {
        double accum = 0;
        for (int i_in = 0; i_in < l.in; i_in++)
        {
            accum += l.weights[i_out * l.in + i_in] * (inputs[i_in]);
        }
        outputs[i_out] = accum + l.biases[i_out];
    }
}

image parse_line(char *line)
{
    image result;
    char *token = strtok(line, ",");
    result.label = atoi(token);
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        result.expected[i] = 0;
    }

    result.expected[result.label] = 1;

    for (int i = 0; i < INPUT_SIZE; i++)
    {
        token = strtok(NULL, ",");
        result.img[i] = ((double)atoi(token)) / 255;
    }
    return result;
}

void print_image_data(image d)
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

void print_double_arr(size_t print_width, size_t size, double *arr)
{
    for (int i = 0; i < size; i++)
    {
        printf("%+012.5lf, ", arr[i]);
        if (i % print_width == (print_width - 1) && i + 1 < size)
        {
            printf("\n");
        }
    }
}

void softmax(int size, double *inputs, double *outputs)
{
    double *e_arr = ass_malloc(sizeof(double) * size);
    double accum = 0;

    for (int i = 0; i < size; i++)
    {
        e_arr[i] = exp(inputs[i]);
        accum += e_arr[i];
    }

    for (int i = 0; i < size; i++)
    {
        outputs[i] = e_arr[i] / accum;
    }
    ass_free(e_arr);
}

// return (x > 0) ? x : 0;
double relu(double x)
{
    return (x > 0) ? x : 0;
}

// return x > 0;
double derivative_of_relu(double x)
{
    return x > 0;
}

// return 1.0 / (1.0 + exp(-1 * x));
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-1 * x));
}

// return sigmoid(x) * (1 - sigmoid(x));
double derivative_of_sigmoid(double x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

int main(int argc, char const *argv[])
{
    srand(420);

    image *data = ass_malloc(sizeof(image) * TRAINING_DATA_AMOUNT);

    // read data from csv
    {
        printf("START\n");
        const char *TRAIN_DATA_PATH = "mnist_train.csv";
        printf("Opening file: %s\n", TRAIN_DATA_PATH);
        FILE *fptr = fopen(TRAIN_DATA_PATH, "r");

        const int file_buffer_size = 4 * INPUT_SIZE + 2; // make the buffer just big enough to hold a single line of well formatted data: input size times 4 for "xxx,", plus two, for the label at the front "x,"
        char file_buffer[file_buffer_size];

        // the first line (the one that explains the layout) it actually longer than the buffer size, so we read twice to read past it.
        fgets(file_buffer, file_buffer_size, fptr);
        fgets(file_buffer, file_buffer_size, fptr);

        printf("\n");
        for (int i = 0; i < TRAINING_DATA_AMOUNT; i++)
        {
            char *line = fgets(file_buffer, file_buffer_size, fptr);
            assert(line != NULL); // Ran out of lines when reading training data, make sure TRAINING_DATA_AMOUNT <= the amount of lines of atual data in the csv
            data[i] = parse_line(line);
        }
        fclose(fptr);
    }

    double (*activation)(double);
    double (*activation_derivative)(double);

    activation = sigmoid;
    activation_derivative = derivative_of_sigmoid;

    layer layers[] = {layer_new(INPUT_SIZE, 128), layer_new(128, 64), layer_new(64, OUTPUT_SIZE)};
    const size_t layer_amount = sizeof(layers) / sizeof(layers[0]);
    DEBUG("%d\n", layer_amount);

    double *actual_results[layer_amount + 1]; // the actual stack allocated array for the results of one training example (including the input data)
    double **results = &(actual_results[1]);  // offset the indexing of results by one, basically creating a "-1" index, this way the indexing still matches the layers[]
    // results[-1] doesn't need a new allocated buffer, since it's just gonna be pointing to already allocated memory in data[]
    for (int layer = 0; layer < layer_amount; layer++)
    {
        results[layer] = ass_malloc(sizeof(double) * layers[layer].out);
    }

    const size_t batch_amount = TRAINING_DATA_AMOUNT / BATCH_SIZE;
    assert(batch_amount * BATCH_SIZE == TRAINING_DATA_AMOUNT); // DATA_AMOUNT should be divisble by BATCH_SIZE
    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        if (epoch % 100 == 0)
        {
            DEBUG("%d\n", epoch);
        }

        shuffle_arr(TRAINING_DATA_AMOUNT, sizeof(data[0]), data);

        for (int batch = 0; batch < batch_amount; batch++)
        {

            for (int training = 0; training < BATCH_SIZE; training++)
            {

                // forward propegate
                results[-1] = data[batch * BATCH_SIZE + training].img; // the "output" of the input "layer" is just the input data
                for (int layer = 0; layer < layer_amount; layer++)
                {
                    layer_apply(layers[layer], results[layer - 1], results[layer]); // apply the dense layer
                    for (int output = 0; output < layers[layer].out; output++)      // apply the activation
                    {
                        results[layer][output] = activation(results[layer][output]);
                    }
                }

                // setup for backpropagation
                double *dcost_dout = ass_calloc(sizeof(double) * layers[layer_amount - 1].out);

                // compute derivative of error with respect to network's output
                // ie. for the 'euclidian distance' cost function, (output  - expected)^2, this would be 2(output - expected) ??? (output - expected)
                for (int out = 0; out < layers[layer_amount - 1].out; out++)
                {
                    dcost_dout[out] = (results[layer_amount - 1][out] - data[batch * BATCH_SIZE + training].expected[out]);
                }

                // Backpropagate
                double eta = 0.15;
                double *next_dcost_dout;
                for (int layer = layer_amount - 1; layer >= 0; layer--)
                {
                    /*
                     * side note:
                     * we're being kinda wastefull here to help generalize, since we're allocating a big array for the dcost_dout of the input values,
                     * values for it, just to throw them out since that isn't a real layer. Definetly a possible place to optimize
                     * if we're fine with introducing more hard coded "edge cases" such as the first and last loop
                     */
                    next_dcost_dout = ass_calloc(sizeof(double) * layers[layer].in); // alloc new array according to the previous layers (next in the backpropagation, since we're propagating backwards) output, aka this layers input

                    for (int out = 0; out < layers[layer].out; out++)
                    {
                        double dout_dz = activation_derivative(results[layer][out]); //! <- only real diff I can see is that in the example that works, this uses the "Out" value after activation instead of the "z" value before activation, so why does 3B1B say it's the derivative of the activation of z???
                        for (int input = 0; input < layers[layer].in; input++)
                        {
                            double dz_dw = results[layer - 1][input];
                            next_dcost_dout[input] += (dcost_dout[out] * dout_dz * layers[layer].weights[out * layers[layer].in + input]); // uses old weight, so has to come before adjustment
                            layers[layer].weights[out * layers[layer].in + input] -= eta * dcost_dout[out] * dout_dz * dz_dw;              // adjust weight
                        }
                        layers[layer].biases[out] -= eta * dcost_dout[out] * dout_dz; // adjust bias
                    }

                    ass_free(dcost_dout);
                    dcost_dout = next_dcost_dout; // reassign next_dcost_dout to dcost_dout before going to prev_layer
                }

                ass_free(next_dcost_dout);
            }
        }
    }

    // print layer weights
    // for (int layer = 0; layer < layer_amount; layer++)
    // {
    // DEBUG("%d:\n", layer);
    // print_double_arr(layers[layer].in, layers[layer].in * layers[layer].out, layers[layer].weights);
    // printf("\n");
    // }

    // print examples to look at
    for (int printed_example = PRINTED_EXAMPLE; printed_example < PRINTED_EXAMPLE_AMOUNT; printed_example++)
    {
        printf("Using model on data nr. (%d):\n", printed_example);
        print_image_data(data[printed_example]); // print the example image

        // forward propegate
        results[-1] = data[printed_example].img;
        for (int layer = 0; layer < layer_amount; layer++)
        {
            {
                layer_apply(layers[layer], results[layer - 1], results[layer]);
                for (int output = 0; output < layers[layer].out; output++)
                {
                    results[layer][output] = activation(results[layer][output]);
                }
            }
        }

        // softmax((layers[layer_amount - 1].out, results[layer_amount - 1], results[layer_amount - 1]);

        printf("Results data nr. (%d):\n", printed_example);
        print_double_arr(layers[layer_amount - 1].out, layers[layer_amount - 1].out, results[layer_amount - 1]);
        printf("\n____________________________________\n");
    }

    // clean up result buffers
    // results[-1] doesn't need to be cleaned, as it's just a pointer to part of the data[] array
    for (int result = 0; result < layer_amount; result++)
    {
        ass_free(results[result]);
    }

    // clean up layers
    for (int layer = 0; layer < layer_amount; layer++)
    {
        layer_del(layers[layer]);
    }

    ass_free(data);
    DEBUG("%d\n", alloc_counter);
    return 0;
}
