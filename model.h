#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>

#define INPUT_SIZE   784
#define FC1_OUT      16
#define OUTPUT_SIZE  10

void run_inference(const volatile int8_t *input, volatile int8_t *output);

#endif