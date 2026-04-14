#include "model.h"
#include "model_weights.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include <string.h>

// Buffers
static int8_t fc1_out[FC1_OUT];
static int8_t relu_out[FC1_OUT];

// Context buffer
static int16_t buffer[1024];

void run_inference(int8_t *input, int8_t *output)
{
    cmsis_nn_context ctx;
    ctx.buf = buffer;
    ctx.size = sizeof(buffer);

    // FC1 parameters
    cmsis_nn_fc_params fc1_params;
    fc1_params.input_offset = 0;
    fc1_params.filter_offset = 0;
    fc1_params.output_offset = 0;
    fc1_params.activation.min = -128;
    fc1_params.activation.max = 127;

    //cmsis_nn_per_tensor_quant_params fc1_quant={1073741824, 30};
    cmsis_nn_per_tensor_quant_params fc1_quant = {1073741824, 7}; 

    cmsis_nn_dims input_dims = {1, 1, 1, INPUT_SIZE};
    cmsis_nn_dims fc1_w_dims = {FC1_OUT, INPUT_SIZE, 1, 1};
    //cmsis_nn_dims bias_dims = {FC1_OUT, 1, 1, 1};
    cmsis_nn_dims fc1_bias_dims = {1, 1, 1, FC1_OUT};
    cmsis_nn_dims fc2_bias_dims = {1, 1, 1, OUTPUT_SIZE};
    cmsis_nn_dims fc1_out_dims = {1, 1, 1, FC1_OUT};

    arm_fully_connected_s8(
        &ctx,
        &fc1_params,
        &fc1_quant,
        &input_dims,
        input,
        &fc1_w_dims,
        (const int8_t *)fc1_weights,
        //&bias_dims,
        &fc1_bias_dims,
        fc1_bias,
        &fc1_out_dims,
        fc1_out
    );

    // ReLU
    for (int i = 0; i < FC1_OUT; i++)
    {
        relu_out[i] = (fc1_out[i] > 0) ? fc1_out[i] : 0;
    }

    // FC2 parameters
    cmsis_nn_fc_params fc2_params;
    fc2_params.input_offset = 0;
    fc2_params.filter_offset = 0;
    fc2_params.output_offset = 0;
    fc2_params.activation.min = -128;
    fc2_params.activation.max = 127;

    cmsis_nn_per_tensor_quant_params fc2_quant;
    fc2_quant.multiplier = 1073741824;
    //fc2_quant.shift = 30;
    fc2_quant.shift = 8;

    cmsis_nn_dims fc2_w_dims = {OUTPUT_SIZE, FC1_OUT, 1, 1};
    cmsis_nn_dims fc2_out_dims = {1, 1, 1, OUTPUT_SIZE};

    arm_fully_connected_s8(
        &ctx,
        &fc2_params,
        &fc2_quant,
        &fc1_out_dims,
        relu_out,
        &fc2_w_dims,
        (const int8_t *)fc2_weights,
        //&bias_dims,
        &fc2_bias_dims,
        fc2_bias,
        &fc2_out_dims,
        output
    );
}