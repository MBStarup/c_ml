#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>

#define DATA_SIZE 784
#define DATA_WIDTH 28
#define DATA_HEIGHT 28
#define DATA_AMOUNT 1000

typedef struct
{
    int label;
    float img[DATA_SIZE];
} image;

typedef struct
{
    int in;
    int out;
    float *weights;
    float *biases;
    float (*func)(float);
} layer;

void randomize_float_arr(float *arr, int size, float min, float max)
{
    for (size_t i = 0; i < size; i++)
    {
        arr[i] = min + (((float)rand()) / ((float)RAND_MAX)) * (max - min);
    }
}

layer layer_new(int in, int out, float (*func)(float))
{
    layer res;

    res.in = in;
    res.out = out;

    res.weights = malloc(in * out * sizeof(res.in));
    assert(res.weights != NULL);
    randomize_float_arr(res.weights, in * out, 0, 1);

    res.biases = malloc(in * out * sizeof(res.in));
    assert(res.biases != NULL);
    randomize_float_arr(res.biases, in * out, 0, 1);

    res.func = func;

    return res;
}

void layer_del(layer l)
{
    free(l.biases);
    free(l.weights);
}

// Assumes the size of inputs matches the size of l.in
// if greater, will cut off remaning inputs
// if smaller will potentially segfault
// Assumes the size of outputs matches the size of l.out
// writes results to outputs
void layer_apply(layer l, float *inputs, float *outputs)
{
    for (size_t i_out = 0; i_out < l.out; i_out++)
    {
        int accum = 0;
        for (size_t i_in = 0; i_in < l.in; i_in++)
        {
            accum += l.weights[i_out * l.in + i_in] * inputs[i_in] + l.biases[i_out * l.in + i_in];
        }
        outputs[i_out] = l.func(accum);
    }
}

image parse_line(char *line)
{
    image result;
    char *token = strtok(line, ",");
    result.label = atoi(token);

    for (int i = 0; i < DATA_SIZE; i++)
    {
        token = strtok(NULL, ",");
        result.img[i] = (float)atoi(token);
    }
    return result;
}

void print_data(image d)
{
    printf("Label: %d\n", d.label);
    for (int i = 0; i < DATA_WIDTH; i++)
    {
        for (int j = 0; j < DATA_HEIGHT; j++)
        {
            d.img[i * DATA_WIDTH + j] > 100 ? printf("1") : printf("0");
        }
        printf("\n");
    }
}

float relu(float x)
{
    return (x > 0) ? x : 0;
}

float sigmoid(float x)
{
    return 1.0 / (1.0 + exp(-1 * x));
}

int main(int argc, char const *argv[])
{
    printf("START\n");
    const char *TRAIN_DATA_PATH = "mnist_train.csv";
    const char *TEST_DATA_PATH = "mnist_test.csv";

    FILE *fptr = fopen(TRAIN_DATA_PATH, "r");

    const int buffer_size = 4 * DATA_SIZE + 2;
    char buffer[buffer_size];

    fgets(buffer, buffer_size, fptr);
    fgets(buffer, buffer_size, fptr);

    image *data = malloc(sizeof(image) * DATA_AMOUNT);
    for (size_t i = 0; i < DATA_AMOUNT; i++)
    {
        char *line = fgets(buffer, buffer_size, fptr);
        assert(line != NULL);
        data[i] = parse_line(line);
    }
    fclose(fptr);

    srand(time(0));

    layer first_layer = layer_new(DATA_SIZE, 30, sigmoid);
    layer hidden_layer_1 = layer_new(30, 24, sigmoid);
    layer hidden_layer_2 = layer_new(24, 50, sigmoid);
    layer last_layer = layer_new(50, 10, sigmoid);

    float *first_result = malloc(sizeof(float) * first_layer.out);
    float *hidden_result_1 = malloc(sizeof(float) * hidden_layer_1.out);
    float *hidden_result_2 = malloc(sizeof(float) * hidden_layer_2.out);
    float *last_result = malloc(sizeof(float) * last_layer.out);

    layer_apply(first_layer, data[0].img, first_result);
    layer_apply(hidden_layer_1, first_result, hidden_result_1);
    layer_apply(hidden_layer_2, hidden_result_1, hidden_result_2);
    layer_apply(last_layer, hidden_result_2, last_result);

    layer_del(first_layer);
    layer_del(hidden_layer_1);
    layer_del(hidden_layer_2);
    layer_del(last_layer);

    for (size_t i = 0; i < last_layer.out; i++)
    {
        printf("%f, ", last_result[i]);
    }

    // printf("%s", line1);

    free(data);
    free(first_result);
    free(hidden_result_1);
    free(hidden_result_2);
    free(last_result);
    printf("END\n");
    return 0;
}