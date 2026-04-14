#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>

#define INPUT_SIZE   784
#define FC1_OUT      64
#define OUTPUT_SIZE  10

void run_inference(int8_t *input, int8_t *output);

#endif