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

    free(temp);
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
    free(e_arr);
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

    layer I_H1 = layer_new(INPUT_SIZE, 128);
    layer H1_O = layer_new(128, OUTPUT_SIZE);

    double *I_H1_results = ass_malloc(sizeof(double) * I_H1.out);
    double *H1_O_results = ass_malloc(sizeof(double) * H1_O.out);
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

                prev_layer_gradient = ass_malloc(sizeof(double) * H1_O.in);
                gradient = ass_calloc(sizeof(double) * H1_O.out);

                // forward propegate
                // I_H1
                layer_apply(I_H1, data[batch * BATCH_SIZE + training].img, I_H1_results);
                for (size_t output = 0; output < I_H1.out; output++)
                {
                    I_H1_results[output] = activation(I_H1_results[output]);
                }

                // H1_O
                layer_apply(H1_O, I_H1_results, H1_O_results);
                for (size_t output = 0; output < H1_O.out; output++)
                {
                    H1_O_results[output] = activation(H1_O_results[output]);
                }

                // compute derivative of error with respect to network's output
                for (int out = 0; out < H1_O.out; out++)
                {
                    gradient[out] = (H1_O_results[out] - data[batch * BATCH_SIZE + training].expected[out]);
                }

                // Backpropegate
                double eta = 0.15;
                // H1_O
                for (int out = 0; out < H1_O.out; out++)
                {
                    gradient[out] *= activation_derivative(H1_O_results[out]);
                }
                for (int input = 0; input < H1_O.in; input++)
                {
                    double g = 0.0;
                    for (int out = 0; out < H1_O.out; out++)
                    {
                        g += (gradient[out] * H1_O.weights[out * H1_O.in + input]);
                    }
                    prev_layer_gradient[input] = g;
                }

                // change weights using gradient
                for (int out = 0; out < H1_O.out; out++)
                {
                    for (int input = 0; input < H1_O.in; input++)
                    {
                        H1_O.weights[out * H1_O.in + input] -= (eta * gradient[out] * I_H1_results[input]);
                    }
                    H1_O.biases[H1_O.in] -= eta * gradient[out];
                }

                free(gradient);                 // free old graident
                gradient = prev_layer_gradient; // reassign prev_layer_gradient to gradient before going to prev_layer
                prev_layer_gradient = ass_malloc(sizeof(double) * I_H1.in);

                // I_H1
                for (int out = 0; out < I_H1.out; out++)
                {
                    gradient[out] *= activation_derivative(I_H1_results[out]);
                }
                for (int input = 0; input < I_H1.in; input++)
                {
                    double g = 0.0;
                    for (int out = 0; out < I_H1.out; out++)
                    {
                        g += (gradient[out] * I_H1.weights[out * I_H1.in + input]);
                    }
                    prev_layer_gradient[input] = g;
                }

                // change weights using gradient
                for (int out = 0; out < I_H1.out; out++)
                {
                    for (int input = 0; input < I_H1.in; input++)
                    {
                        I_H1.weights[out * I_H1.in + input] -= (eta * gradient[out] * data[batch * BATCH_SIZE + training].img[input]);
                    }
                    I_H1.biases[I_H1.in] -= eta * gradient[out];
                }

                free(gradient);
                free(prev_layer_gradient);
            }
        }
    }

    DEBUG(":\n", I_H1);
    print_double_arr(I_H1.in, I_H1.in * I_H1.out, I_H1.weights);
    printf("\n");
    DEBUG(":\n", H1_O);
    print_double_arr(H1_O.in, H1_O.in * H1_O.out, H1_O.weights);
    printf("\n");

    for (size_t printed_example = PRINTED_EXAMPLE; printed_example < PRINTED_EXAMPLE_AMOUNT; printed_example++)
    {
        printf("using model on %d:\n", printed_example);
        print_image_data(data[printed_example]); // print the example image
        // forward propegate
        {
            // I_H1
            {
                layer_apply(I_H1, data[printed_example].img, I_H1_results);
                for (size_t output = 0; output < I_H1.out; output++)
                {
                    I_H1_results[output] = activation(I_H1_results[output]);
                }
            }

            // H1_O
            {
                layer_apply(H1_O, I_H1_results, H1_O_results);
                for (size_t output = 0; output < H1_O.out; output++)
                {
                    H1_O_results[output] = activation(H1_O_results[output]);
                }
            }
        }

        // softmax(H1_O.out, H1_O_results, H1_O_results);

        printf("results (%d):\n", printed_example);
        print_double_arr(H1_O.out, H1_O.out, H1_O_results);
        printf("\n____________________________________\n");
    }

    free(I_H1_results);
    free(H1_O_results);
    free(data);
    return 0;
}
