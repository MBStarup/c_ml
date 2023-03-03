#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>

#define DATA_SIZE 784
#define DATA_WIDTH 28
#define DATA_HEIGHT 28
#define DATA_AMOUNT 500
#define PRINT_NUM 5
#define EPOCHS 500
#define PRINTED_EXAMPLE 17
#define PRINTED_EXAMPLE_AMOUNT 3
#define NO_PRINT

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
    randomize_double_arr(res.weights, in * out, -1, 1);

    res.biases = ass_malloc(sizeof(double) * out);
    randomize_double_arr(res.biases, out, -1, 1);

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
        outputs[i_out] = accum + l.biases[i_out]; //! BIAS
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

void print_image_data(image d)
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

void print_double_arr(size_t size, double *arr)
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

double relu(double x)
{
    return (x > 0) ? x : 0;
}

double derivative_of_relu(double x)
{
    return x > 0;
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-1 * x));
}

double derivative_of_sigmoid(double x)
{
    return sigmoid(x) * (1 - sigmoid(x));
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
    srand(101);
    // srand(time(0));

    model model = model_new("test_model", 2);
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

        model_add(&model, layer_new(DATA_SIZE, 128));
        model_add(&model, layer_new(128, 10));
    }

    //* for every training example, allocate a buffer for the input data, followed by a buffer for the intermediate results between the layers
    double **results = ass_malloc(sizeof(double *) * (DATA_AMOUNT * (model.layer_amount + 1)));
    for (int i = 0; i < DATA_AMOUNT; i++)
    {
        results[i * (model.layer_amount + 1) + 0] = data[i].img; // set the zero'th "result" to the input data
        for (size_t layer = 0; layer < model.layer_amount; layer++)
        {
            results[i * (model.layer_amount + 1) + (layer + 1)] = ass_malloc(sizeof(double) * model.layers[layer].out); // TODO: free
        }

        layer_apply(model.layers[0], data[i].img, results[i * (model.layer_amount + 1) + (0 + 1)]); // layer 0, plus an offset of 1 for the input data at the front
        for (size_t layer = 1; layer < model.layer_amount; layer++)
        {
            layer_apply(model.layers[layer], results[i * (model.layer_amount + 1) + layer], results[i * (model.layer_amount + 1) + layer + 1]); // Apply the layer, use the output from the prev layer as the input
        }
    }

    double **weights_adjustments = ass_malloc(sizeof(double *) * model.layer_amount); // TODO: free
    double **bias_adjustments = ass_malloc(sizeof(double *) * model.layer_amount);    // TODO: free
    for (size_t layer = 0; layer < model.layer_amount; layer++)
    {
        size_t weight_n = model.layers[layer].in * model.layers[layer].out;
        weights_adjustments[layer] = ass_calloc(sizeof(double) * weight_n); // TODO: free

        bias_adjustments[layer] = ass_calloc(sizeof(double) * model.layers[layer].out); // TODO: free
    }
    // Now we have an array of all the adjustments we want to make to the layers' weights, which themselves are an array, initialized to zeros
    // and an array of adjustments to biases
    printf("Starting Training\n\n");
    double alpha = 0.004;
    double eta = 0.0005;
    for (size_t epoch = 0; epoch < EPOCHS; epoch++)
    {
        if (epoch % 100 == 0)
            printf("epoch: %d\n", epoch);
        double one_over_trainig_amount = 1.0 / (double)DATA_AMOUNT;
        for (size_t training = 0; training < DATA_AMOUNT; training++)
        {
            double *expected = data[training].expected;
            for (int layer = model.layer_amount - 1; layer >= 0; layer--)
            {
                double *new_expected = ass_calloc(sizeof(double) * model.layers[layer].in); // TODO: free
                double one_over_input_amount = (double)1.0 / model.layers[layer].in;
                for (size_t output_neuron = 0; output_neuron < model.layers[layer].out; output_neuron++)
                {
                    //* Target - Output =>  results[t*(bs) + 2]    -                                    results[t*(bs) + 1]
                    //*                   (expected[output_neuron] - sigmoid(results[training * (model.layer_amount + 1) + (layer + 1)][output_neuron]))
                    double dout_dz = derivative_of_sigmoid(results[training * (model.layer_amount + 1) + (layer + 1)][output_neuron]);
                    assert(!isnan(dout_dz));

                    double x = results[training * (model.layer_amount + 1) + (layer + 1)][output_neuron];
                    // printf("Sigmoiding: %lf\n", x);
                    double y = sigmoid(x);
                    // printf("y: %lf\n", y);
                    assert(!isnan(y));
                    double z = expected[output_neuron];
                    // printf("z: %lf\n", z);
                    assert(!isnan(z));
                    double dcost_dout = 2 * (y - z);
                    assert(!isnan(dcost_dout));

                    for (size_t input_neuron = 0; input_neuron < model.layers[layer].in; input_neuron++)
                    {
#ifndef NO_PRINT
                        printf("\x1b[1F");
                        printf("epoch: %d/%d | ", epoch + 1, EPOCHS);
                        printf("training: %d/%d | ", training + 1, DATA_AMOUNT);
                        printf("layer: %d/%d | ", layer + 1, model.layer_amount);
                        printf("out_nron: %d/%d | ", output_neuron + 1, model.layers[layer].out);
                        printf("in_nron: %d/%d | ", input_neuron, model.layers[layer].in);
                        printf("\n");
#endif

                        double dz_dw = sigmoid(results[training * (model.layer_amount + 1) + (layer + 1) - 1][input_neuron]);
                        assert(!isnan(dz_dw));

                        // L € {3, 2, 1, 0}

                        // input neuron for current layer = output neuron for layer -1
                        // a^(L - 1)[out_nron] = > sigmoid(results[training * (model.layer_amount + 1) + (layer + 1) - 1][input_neuron]);
                        // sigma''(z ^ (L)) = > derivative_of_sigmoid(results[training * (model.layer_amount + 1) + (layer + 1)][output_neuron]);
                        // 2(a ^ (L)-y) = > 2 * (sigmoid(results[training * (model.layer_amount + 1) + (layer + 1)][output_neuron]) - expected[output_neuron])

                        // DeltaWeightHO[j][k] = eta * Hidden[p][j] * DeltaO[k] + alpha * DeltaWeightHO[j][k];
                        // eta * dz_dw * dout_dz * dcost_dout + alpha * ???

                        double cringe = one_over_trainig_amount * 0.1;

                        //    weights_adjustments[layer][output_neuron * model.layers[layer].in + input_neuron] += one_over_trainig_amount * eta * dcost_dout * dout_dz * dz_dw;
                        weights_adjustments[layer][output_neuron * model.layers[layer].in + input_neuron] += cringe * dcost_dout * dout_dz * dz_dw + alpha * weights_adjustments[layer][output_neuron * model.layers[layer].in + input_neuron];
                        new_expected[input_neuron] += one_over_input_amount * model.layers[layer].weights[output_neuron * model.layers[layer].in + input_neuron] * dcost_dout * dout_dz;
                        // printf("one_over_input_amount: %lf ", one_over_input_amount);
                        // printf("model.layers[layer].weights[output_neuron * model.layers[layer].in + input_neuron]: %lf ", model.layers[layer].weights[output_neuron * model.layers[layer].in + input_neuron]);
                        // printf("dcost_dout: %lf ", dcost_dout);
                        // printf("dout_dz: %lf ", dout_dz);
                        // printf("new_expected[%d]: %lf\n", input_neuron, new_expected[input_neuron]);
                    }
                    // bias_adjustments[layer][output_neuron] += one_over_trainig_amount * dout_dz * dcost_dout; //! BIAS
                    bias_adjustments[layer][output_neuron] += eta * dcost_dout * dout_dz + alpha * bias_adjustments[layer][output_neuron]; //! BIAS
                }
                expected = new_expected;
            }
        }

        // # apply adjustments:
        // for each layer, adjust the weight of each input/output neuron pair by the amount calculated in the weights_adjustments[]
        // for each layer, adjust the bias of each output neuron by the amount calculated in the bias_adjustments[]
        for (size_t layer = 0; layer < model.layer_amount; layer++)
        {
            size_t weight_n = model.layers[layer].in * model.layers[layer].out; // the amount of weights in the layer
            for (size_t output_neuron = 0; output_neuron < model.layers[layer].out; output_neuron++)
            {
                for (size_t input_neuron = 0; input_neuron < model.layers[layer].in; input_neuron++)
                {
                    // printf("Epoch: %d: Adjusting weight %d (%d x %d) in layer %d by %lf\n", epoch + 1, output_neuron * model.layers[layer].in + input_neuron, input_neuron + 1, output_neuron + 1, layer + 1, weights_adjustments[layer][output_neuron * model.layers[layer].in + input_neuron]);
                    model.layers[layer].weights[output_neuron * model.layers[layer].in + input_neuron] += weights_adjustments[layer][output_neuron * model.layers[layer].in + input_neuron];
                }
                // printf("Epoch: %d: Adjusting bias %d in layer %d by %lf\n", epoch + 1, output_neuron + 1, layer + 1, bias_adjustments[layer][output_neuron]);
                model.layers[layer].biases[output_neuron] += bias_adjustments[layer][output_neuron]; //! BIAS
            }

            memset(weights_adjustments[layer], 0, sizeof(double) * weight_n);             // reset the adjustments to zero
            memset(bias_adjustments[layer], 0, sizeof(double) * model.layers[layer].out); // reset the adjustments to zero
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
                final_results[0][output] = sigmoid(final_results[0][output]); // apply activation function
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
                    final_results[layer][output] = sigmoid(final_results[layer][output]); // apply activation function
                }

                // print for debug
                printf("Layer %d results :\n", layer + 1);
                print_double_arr(model.layers[layer].out, final_results[layer]);
                printf("\n------------------\n");
            }

            // apply softmax on the last result to get probability result
            // softmax(model.layers[model.layer_amount - 1].out, final_results[model.layer_amount - 1], final_results[model.layer_amount - 1]);

            print_double_arr(model.layers[model.layer_amount - 1].out, final_results[model.layer_amount - 1]); // print the resulting probability weights for the example

            free(final_results);
        }
    }

    model_del(model);
    free(data);
    printf("END\n");
    return 0;
}