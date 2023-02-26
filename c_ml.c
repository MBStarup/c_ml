#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <unistd.h>

#define DATA_SIZE 784
#define DATA_WIDTH 28
#define DATA_HEIGHT 28
#define DATA_AMOUNT 10000
#define PRINT_NUM 5
#define LAYER_AMOUNT 4
#define EPOCHS 10

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

typedef struct
{
    size_t layer_capacity;
    size_t layer_amount;
    layer *layers;
    char *name;
} model;

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

model model_new(char *name, size_t capacity)
{
    model result;
    result.layer_capacity = capacity;
    result.layer_amount = 0;
    result.name = name;
    result.layers = malloc(sizeof(layer) * capacity);
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
    printf("adding layer %d of %d to model %s\n", model->layer_amount + 1, model->layer_capacity, model->name);
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

void print_data(image d)
{
    printf("Label: %d\n", d.label);
    for (int i = 0; i < DATA_WIDTH; i++)
    {
        for (int j = 0; j < DATA_HEIGHT; j++)
        {
            d.img[i * DATA_WIDTH + j] > 0 ? printf("  ") : printf("[]");
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
// os to hvad med os to
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
    printf("Opening file: %s\n", TRAIN_DATA_PATH);
    FILE *fptr = fopen(TRAIN_DATA_PATH, "r");

    const int buffer_size = 4 * DATA_SIZE + 2;
    char buffer[buffer_size];

    fgets(buffer, buffer_size, fptr);
    fgets(buffer, buffer_size, fptr);

    image *data = malloc(sizeof(image) * DATA_AMOUNT);
    printf("\n");
    for (size_t i = 0; i < DATA_AMOUNT; i++)
    {
        printf("\x1b[1F");
        char *line = fgets(buffer, buffer_size, fptr);
        assert(line != NULL);
        data[i] = parse_line(line);
        printf("Parsing data: %d/%d\n", i + 1, DATA_AMOUNT);
        // fflush(stdin);
        // usleep(10000); // TODO: Remove...
    }
    fclose(fptr);

    // srand(420);
    srand(time(0));

    layer first_layer = layer_new(DATA_SIZE, 5, sigmoid);
    layer hidden_layer_1 = layer_new(5, 3, sigmoid);
    layer hidden_layer_2 = layer_new(3, 7, sigmoid);
    layer last_layer = layer_new(7, 10, sigmoid);

    model model = model_new("test_model", LAYER_AMOUNT);
    model_add(&model, first_layer);
    model_add(&model, hidden_layer_1);
    model_add(&model, hidden_layer_2);
    model_add(&model, last_layer);

    double *results[DATA_AMOUNT * LAYER_AMOUNT];

    for (int i = 0; i < DATA_AMOUNT; i++)
    {
        // Haven't freed very big problem but not really though
        double *first_result = malloc(sizeof(double) * first_layer.out);       // TODO: Is never freed
        double *hidden_result_1 = malloc(sizeof(double) * hidden_layer_1.out); // TODO: Is never freed
        double *hidden_result_2 = malloc(sizeof(double) * hidden_layer_2.out); // TODO: Is never freed
        double *last_result = malloc(sizeof(double) * last_layer.out);         // TODO: Is never freed

        results[i * LAYER_AMOUNT + 0] = first_result;
        results[i * LAYER_AMOUNT + 1] = hidden_result_1;
        results[i * LAYER_AMOUNT + 2] = hidden_result_2;
        results[i * LAYER_AMOUNT + 3] = last_result;

        layer_apply(first_layer, data[i].img, first_result);
        for (size_t layer = 1; layer < model.layer_amount; layer++)
        {
            // Apply the layer, use the output from the prev layer as the input
            layer_apply(model.layers[layer], results[layer - 1], results[layer]);
        }
    }

    // for every piece of training data, do the following
    // start at the last layer, the training data provides an expected value that can be used to calculate local cost
    // use this cost to calcuate the best postitions the previous layer could have been in to minimize this cost ?- with the current weights -?, this will become the expected value for the prev layer
    // then use the cost to calclate the best value for each of the weights if the prev layer was ?- the newly calulated expected value || as it happened to be that training -?
    // apply these || store these to sum up over all the training data at the end

    // biases are probably somewhat the same

    // foreach node in the current layer, calculate differential of the cost function with respect to the each of the previous nodes
    // the differential of the current node (pre-activation func) with respect to the prev output is just the weight
    // the differential of the current node (pre-activation func) with respect to the weight is the prev output
    // once we have the differential of the cost (just squared difference from expected? with respect to the prev nodes, this value will tell you how much you should want the node to be adjusted
    // sum this up, and the prev node value times this adjustment becomes the expected values for the prev layer

    // where do we adjust the weights and biases then?
    // before or after we calculate the expectd values?

    // foreach layer take the size of the ins times the size of the outs to get the number of weigths
    double **weights_adjustments = malloc(model.layer_amount); // TODO: free
    double **bias_adjustments = malloc(model.layer_amount);    // TODO: free
    for (size_t layer = 0; layer < model.layer_amount; layer++)
    {
        size_t weight_n = model.layers[layer].in * model.layers[layer].out;
        weights_adjustments[layer] = malloc(sizeof(double) * weight_n); // TODO: Clean up
        assert(weights_adjustments[layer] != NULL);
        memset(weights_adjustments[layer], 0, sizeof(double) * weight_n);

        bias_adjustments[layer] = malloc(sizeof(double) * model.layers[layer].out); // TODO: Clean up
        assert(bias_adjustments[layer] != NULL);
        memset(bias_adjustments[layer], 0, sizeof(double) * model.layers[layer].out);
    }
    // Now we have an array of all the adjustments we want to make to the layers' weights, which themselves are an array, initialized to zeros
    // and an array of adjustments to biases

    for (size_t epoch = 0; epoch < EPOCHS; epoch++)
    {
        // printf("EPOCH: %d/%d\n", epoch + 1, EPOCHS);

        for (size_t training = 0; training < DATA_AMOUNT; training++)
        {
            // printf("    training: %d/%d\n", training + 1, DATA_AMOUNT);
            for (int layer = model.layer_amount - 1; layer >= 0; layer--)
            {
                // printf("        layer: %d/%d\n", layer + 1, model.layer_amount);
                double *expected = data[training].expected;
                for (size_t output_neuron = 0; output_neuron < model.layers[layer].out; output_neuron++)
                {
                    // printf("            out_nron: %d/%d\n", output_neuron + 1, model.layers[layer].out);
                    double delta = (results[model.layer_amount * training + layer][output_neuron] - expected[output_neuron]); // TODO: make sure the subtraction is in the right order
                    for (size_t input_neuron = 0; input_neuron < model.layers[layer].in; input_neuron++)
                    {
                        // printf("            adjusting weight: %d/%d\n", output_neuron * model.layers[layer].in + input_neuron + 1, model.layers[layer].in * model.layers[layer].out);
                        // printf("                in_nron: %d/%d\n", input_neuron, model.layers[layer].in);
                        // The desired adjustent to the weight between a given pair of output/input for a specific training case is proportial to the difference from the expected output and proportional to the strength of the input
                        // IE. the more output from the prev layer for the given node, the more important it is the weights match, and the more the result is wrong, the more we need to adjust by to fix it
                        // note: we += as we're summing up this desired adjustment for all the test cases
                        // printf("Calculated weight adjustment: %lf", delta * model.layers[layer].weights[output_neuron * model.layers[layer].in + input_neuron]);
                        weights_adjustments[layer][output_neuron * model.layers[layer].in + input_neuron] += delta * model.layers[layer].weights[output_neuron * model.layers[layer].in + input_neuron] * 1; // TODO: add scaling factor for control
                        // printf("Now the summed adjustment is %lf\n", weights_adjustments[layer][output_neuron * model.layers[layer].in + input_neuron]);
                    }
                }
            }
        }

        // apply adjustments
        for (size_t layer = 0; layer < model.layer_amount; layer++)
        {
            size_t weight_n = model.layers[layer].in * model.layers[layer].out;
            // TODO: biases
            for (size_t output_neuron = 0; output_neuron < model.layers[layer].out; output_neuron++)
            {
                for (size_t input_neuron = 0; input_neuron < model.layers[layer].in; input_neuron++)
                {
                    // printf("Epoch: %d: Adjusting weight %d in layer %d by %lf\n", epoch + 1, output_neuron * model.layers[layer].in + input_neuron, layer + 1, weights_adjustments[layer][output_neuron * model.layers[layer].in + input_neuron]);
                    model.layers[layer].weights[output_neuron * model.layers[layer].in + input_neuron] += weights_adjustments[layer][output_neuron * model.layers[layer].in + input_neuron];
                }
                // printf("Adjusting bias %d in layer %d by %lf\n", output_neuron + 1, layer + 1, bias_adjustments[layer][output_neuron]);
                model.layers[layer].biases[output_neuron] += bias_adjustments[layer][output_neuron];
            }

            memset(weights_adjustments[layer], 0, sizeof(double) * weight_n);
            memset(bias_adjustments[layer], 0, sizeof(double) * model.layers[layer].out);
        }
    }

    // use the trained model
    {

        double *results[LAYER_AMOUNT];

        // Haven't freed very big problem but not really though
        double *first_result = malloc(sizeof(double) * first_layer.out);       // TODO: Is never freed
        double *hidden_result_1 = malloc(sizeof(double) * hidden_layer_1.out); // TODO: Is never freed
        double *hidden_result_2 = malloc(sizeof(double) * hidden_layer_2.out); // TODO: Is never freed
        double *last_result = malloc(sizeof(double) * last_layer.out);         // TODO: Is never freed

        results[0] = first_result;
        results[1] = hidden_result_1;
        results[2] = hidden_result_2;
        results[3] = last_result;

        layer_apply(first_layer, data[69].img, first_result);
        for (size_t layer = 1; layer < model.layer_amount; layer++)
        {
            // Apply the layer, use the output from the prev layer as the input
            layer_apply(model.layers[layer], results[layer - 1], results[layer]);
        }

        layer_del(first_layer);
        layer_del(hidden_layer_1);
        layer_del(hidden_layer_2);
        layer_del(last_layer);

        softmax(last_layer.out, last_result, last_result);
        printf("sum: %lf \n", sum(last_layer.out, last_result));

        print_data(data[69]);

        for (size_t i = 0; i < last_layer.out; i++)
        {
            printf("%+012.5lf, ", last_result[i]);
            if (i % PRINT_NUM == (PRINT_NUM - 1))
            {
                printf("\n");
            }
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