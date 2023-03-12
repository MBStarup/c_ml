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
    for (size_t i = 0; i < size; i++)
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

image parse_line(char *line)
{
    image result;
    char *token = strtok(line, ",");
    result.label = atoi(token);
    for (size_t i = 0; i < OUTPUT_SIZE; i++)
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
    for (size_t i = 0; i < size; i++)
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

    for (size_t i = 0; i < size; i++)
    {
        e_arr[i] = exp(inputs[i]);
        accum += e_arr[i];
    }

    for (size_t i = 0; i < size; i++)
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

    {
        printf("START\n");
        const char *TRAIN_DATA_PATH = "mnist_train.csv";
        printf("Opening file: %s\n", TRAIN_DATA_PATH);
        FILE *fptr = fopen(TRAIN_DATA_PATH, "r");

        const int file_buffer_size = 4 * INPUT_SIZE + 2;
        char file_buffer[file_buffer_size];

        fgets(file_buffer, file_buffer_size, fptr);
        fgets(file_buffer, file_buffer_size, fptr);

        printf("\n");
        for (size_t i = 0; i < TRAINING_DATA_AMOUNT; i++)
        {
            char *line = fgets(file_buffer, file_buffer_size, fptr);
            assert(line != NULL);
            data[i] = parse_line(line);
        }
        fclose(fptr);
    }

    double (*activation)(double);
    double (*activation_derivative)(double);

    activation = sigmoid;
    activation_derivative = derivative_of_sigmoid;

    layer layers[] = {layer_new(INPUT_SIZE, 128), layer_new(128, OUTPUT_SIZE)};
    const size_t layer_amount = sizeof(layers) / sizeof(layers[0]);
    DEBUG("%d\n", layer_amount);

    double *results[layer_amount];
    for (size_t layer = 0; layer < layer_amount; layer++)
    {
        results[layer] = ass_malloc(sizeof(double) * layers[layer].out);
    }
    const size_t result_amount = sizeof(results) / sizeof(results[0]);
    DEBUG("%d\n", result_amount);

    assert(layer_amount == result_amount);

    double *prev_layer_gradient;
    double *gradient;
    const size_t batch_amount = TRAINING_DATA_AMOUNT / BATCH_SIZE;
    assert(batch_amount * BATCH_SIZE == TRAINING_DATA_AMOUNT); // DATA_AMOUNT shoudl be divisble by BATCH_SIZE
    for (size_t epoch = 0; epoch < EPOCHS; epoch++)
    {
        if (epoch % 100 == 0)
        {
            DEBUG("%d\n", epoch);
        }

        shuffle_arr(TRAINING_DATA_AMOUNT, sizeof(data[0]), data);

        for (size_t batch = 0; batch < batch_amount; batch++)
        {

            for (size_t training = 0; training < BATCH_SIZE; training++)
            {

                // forward propegate
                double *input_array;
                double *output_array = data[batch * BATCH_SIZE + training].img; // the "output" of the input "layer"
                for (size_t layer = 0; layer < layer_amount; layer++)
                {
                    input_array = output_array; // the input is the previous layers output
                    output_array = results[layer];
                    layer_apply(layers[layer], input_array, output_array);        // apply the dense layer
                    for (size_t output = 0; output < layers[layer].out; output++) // apply the activation
                    {
                        output_array[output] = activation(output_array[output]);
                    }
                }

                // setup for backpropagation
                gradient = ass_calloc(sizeof(double) * layers[layer_amount - 1].out);
                prev_layer_gradient = ass_malloc(sizeof(double) * layers[layer_amount - 1].in);

                // compute derivative of error with respect to network's output
                // ie. for the 'euclidian distance' cost function, (output  - expected)^2, this would be 2(output - expected) ∝ (output - expected)
                for (int out = 0; out < layers[layer_amount - 1].out; out++)
                {
                    gradient[out] = (results[result_amount - 1][out] - data[batch * BATCH_SIZE + training].expected[out]);
                }

                // Backpropagate
                double eta = 0.15;
                size_t index;
                // H1_O
                for (size_t layer = layer_amount - 1; layer >= 1; layer--)
                {
                    for (int out = 0; out < layers[layer].out; out++)
                    {
                        gradient[out] *= activation_derivative(results[layer][out]);
                    }
                    for (int input = 0; input < layers[layer].in; input++)
                    {
                        double g = 0.0;
                        for (int out = 0; out < layers[layer].out; out++)
                        {
                            g += (gradient[out] * layers[layer].weights[out * layers[layer].in + input]);
                        }
                        prev_layer_gradient[input] = g;
                    }

                    // change weights using gradient
                    for (int out = 0; out < layers[layer].out; out++)
                    {
                        for (int input = 0; input < layers[layer].in; input++)
                        {
                            layers[layer].weights[out * layers[layer].in + input] -= (eta * gradient[out] * results[layer - 1][input]);
                        }
                        layers[layer].biases[out] -= eta * gradient[out];
                    }

                    ass_free(gradient);                                                      // free old graident
                    gradient = prev_layer_gradient;                                          // reassign prev_layer_gradient to gradient before going to prev_layer
                    prev_layer_gradient = ass_malloc(sizeof(double) * layers[layer - 1].in); // alloc new array according to the layer-1'th layers input count
                }

                // Last layer needs special treatment since the input can't be generalized as results[index-1], since it's not the result of a layer
                index = 0;
                for (int out = 0; out < layers[index].out; out++)
                {
                    gradient[out] *= activation_derivative(results[index][out]);
                }
                for (int input = 0; input < layers[index].in; input++)
                {
                    double g = 0.0;
                    for (int out = 0; out < layers[index].out; out++)
                    {
                        g += (gradient[out] * layers[index].weights[out * layers[index].in + input]);
                    }
                    prev_layer_gradient[input] = g;
                }

                // change weights using gradient
                for (int out = 0; out < layers[index].out; out++)
                {
                    for (int input = 0; input < layers[index].in; input++)
                    {
                        layers[index].weights[out * layers[index].in + input] -= (eta * gradient[out] * data[batch * BATCH_SIZE + training].img[input]);
                    }
                    layers[index].biases[out] -= eta * gradient[out];
                }

                ass_free(gradient);
                ass_free(prev_layer_gradient);
            }
        }
    }

    for (size_t layer = 0; layer < layer_amount; layer++)
    {
        DEBUG("%d:\n", layer);
        print_double_arr(layers[layer].in, layers[layer].in * layers[layer].out, layers[layer].weights);
        printf("\n");
    }

    for (size_t printed_example = PRINTED_EXAMPLE; printed_example < PRINTED_EXAMPLE_AMOUNT; printed_example++)
    {
        printf("using model on %d:\n", printed_example);
        print_image_data(data[printed_example]); // print the example image

        // forward propegate
        layer_apply(layers[0], data[printed_example].img, results[0]);
        for (size_t output = 0; output < layers[0].out; output++)
        {
            results[0][output] = activation(results[0][output]);
        }
        for (size_t layer = 1; layer < layer_amount; layer++)
        {
            {
                layer_apply(layers[layer], results[layer - 1], results[layer]);
                for (size_t output = 0; output < layers[layer].out; output++)
                {
                    results[layer][output] = activation(results[layer][output]);
                }
            }
        }

        // softmax((layers[layer_amount - 1].out, results[layer_amount - 1], results[layer_amount - 1]);

        printf("results (%d):\n", printed_example);
        print_double_arr(layers[layer_amount - 1].out, layers[layer_amount - 1].out, results[layer_amount - 1]);
        printf("\n____________________________________\n");
    }

    // clean up result buffers
    for (size_t layer = 0; layer < result_amount; layer++)
    {
        ass_free(results[layer]);
    }

    // clean up layers
    for (size_t layer = 0; layer < layer_amount; layer++)
    {
        layer_del(layers[layer]);
    }

    ass_free(data);
    DEBUG("%d\n", alloc_counter);
    return 0;
}
