#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>

#define DATA_SIZE 784
#define DATA_WIDTH 28
#define DATA_HEIGHT 28
#define DATA_AMOUNT 100
#define PRINT_NUM 5
#define LAYER_AMOUNT 4

typedef struct
{
    int label;
    double img[DATA_SIZE];
    double expected[10];
} image;

typedef struct
{
    int in;
    int out;
    double *weights;
    double *biases;
    double (*func)(double);
} layer;

void randomize_double_arr(double *arr, int size, double min, double max)
{
    for (size_t i = 0; i < size; i++)
    {
        arr[i] = min + (((double)rand()) / ((double)RAND_MAX)) * (max - min);
    }
}

void softmax(int size, double *inputs, double *outputs)
{
    double *e_arr = malloc(sizeof(double) * size);
    assert(e_arr != NULL);
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

layer layer_new(int in, int out, double (*func)(double))
{
    layer res;

    res.in = in;
    res.out = out;

    res.weights = malloc(in * out * sizeof(double));
    assert(res.weights != NULL);
    randomize_double_arr(res.weights, in * out, -1, 1);

    res.biases = malloc(out * sizeof(double));
    assert(res.biases != NULL);
    randomize_double_arr(res.biases, out, -1, 1);

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
void layer_apply(layer l, double *inputs, double *outputs)
{
    for (size_t i_out = 0; i_out < l.out; i_out++)
    {
        double accum = 0;
        for (size_t i_in = 0; i_in < l.in; i_in++)
        {
            accum += l.weights[i_out * l.in + i_in] * inputs[i_in] + l.biases[i_out];
        }
        outputs[i_out] = l.func(accum);
    }
}

image parse_line(char *line)
{
    image result;
    char *token = strtok(line, ",");
    result.label = atoi(token);
    for (size_t i = 0; i < 10; i++)
    {
        result.expected = 0;
    }

    result.expected[label] = 1;

    for (int i = 0; i < DATA_SIZE; i++)
    {
        token = strtok(NULL, ",");
        result.img[i] = ((double)atoi(token)) / 255;
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
            d.img[i * DATA_WIDTH + j] > 0 ? printf("1") : printf("0");
        }
        printf("\n");
    }
}

double relu(double x)
{
    return (x > 0) ? x : 0;
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-1 * x));
}

double x(double x)
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
    printf("START\n");
    const char *TRAIN_DATA_PATH = "mnist_train.csv";
    const char *TEST_DATA_PATH = "mnist_test.csv";
    printf("Here :) \n");
    FILE *fptr = fopen(TRAIN_DATA_PATH, "r");
    printf("Here :) \n");

    const int buffer_size = 4 * DATA_SIZE + 2;
    char buffer[buffer_size];
    printf("Here :) \n");

    fgets(buffer, buffer_size, fptr);
    fgets(buffer, buffer_size, fptr);

    printf("Here :) \n");
    image *data = malloc(sizeof(image) * DATA_AMOUNT);
    for (size_t i = 0; i < DATA_AMOUNT; i++)
    {
        char *line = fgets(buffer, buffer_size, fptr);
        assert(line != NULL);
        data[i] = parse_line(line);
    }
    fclose(fptr);

    printf("Here :) \n");
    srand(420);
    // srand(time(0));
    printf("Here1 :) \n");

    layer first_layer = layer_new(DATA_SIZE, 30, relu);
    layer hidden_layer_1 = layer_new(30, 24, relu);
    layer hidden_layer_2 = layer_new(24, 50, relu);
    layer last_layer = layer_new(50, 10, relu);

    layer layers[LAYER_AMOUNT] = {first_layer, hidden_layer_1, hidden_layer_2, last_layer};
    double *results[DATA_AMOUNT * LAYER_AMOUNT];

    for (int i = 0; i < DATA_AMOUNT; i++)
    {
        // Haven't freed very big problem but not really though
        double *first_result = malloc(sizeof(double) * first_layer.out);
        double *hidden_result_1 = malloc(sizeof(double) * hidden_layer_1.out);
        double *hidden_result_2 = malloc(sizeof(double) * hidden_layer_2.out);
        double *last_result = malloc(sizeof(double) * last_layer.out);

        layer_apply(first_layer, data[i].img, first_result);
        layer_apply(hidden_layer_1, first_result, hidden_result_1);
        layer_apply(hidden_layer_2, hidden_result_1, hidden_result_2);
        layer_apply(last_layer, hidden_result_2, last_result);

        results[i * LAYER_AMOUNT + 0] = first_result;
        results[i * LAYER_AMOUNT + 1] = hidden_result_1;
        results[i * LAYER_AMOUNT + 2] = hidden_result_2;
        results[i * LAYER_AMOUNT + 3] = last_result;
    }

    for (size_t training = 0; training < DATA_AMOUNT; training++)
    {
        for (size_t layer = LAYER_AMOUNT - 1; layer >= 0; layer--)
        {
            double *expected = data[training].expected;
            for (size_t output_neuron = 0; output_neuron < layers[layer].out; output_neuron++)
            {
                double delta = (results[LAYER_AMOUNT * training + layer][output_neuron] - expected[output_neuron]);
                for (size_t prev_layer_output_neuron = 0; prev_layer_output_neuron < layers[layer].in; prev_layer_output_neuron++)
                {
                }

                double cost_scalar_irtpl = ;
            }
        }
    }

    layer_del(first_layer);
    layer_del(hidden_layer_1);
    layer_del(hidden_layer_2);
    layer_del(last_layer);

    // softmax(last_layer.out, last_result, last_result);
    printf("sum: %lf \n", sum(last_layer.out, last_result));

    for (size_t i = 0; i < last_layer.out; i++)
    {
        printf("%+012.5lf, ", last_result[i]);
        if (i % PRINT_NUM == (PRINT_NUM - 1))
        {
            printf("\n");
        }
    }

    // for (size_t j = 0; j < first_layer.out; j++)
    // {
    //     for (size_t i = 0; i < first_layer.in; i++)
    //     {
    //         printf("%+f, ", first_layer.weights[j*first_layer.in+i]);
    //     }
    //     printf("\n");
    // }

    // printf("%s", line1);

    free(data);
    printf("END\n");
    return 0;
}