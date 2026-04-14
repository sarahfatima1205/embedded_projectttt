#include "stm32f4xx_hal.h"
#include "model.h"
#include "test_sample.h"

// Debug variables
volatile int8_t debug_input[INPUT_SIZE];
volatile int8_t debug_output[OUTPUT_SIZE];
volatile int max_idx = 0;

int main(void)
{
    HAL_Init();

    // Example input (replace later with real MNIST sample)
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        //debug_input[i] = (i % 2 == 0) ? 10 : -10;
        debug_input[i] = test_sample[i];
    }

    run_inference(debug_input, debug_output);

    // Find predicted digit
    max_idx = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++)
    {
        if (debug_output[i] > debug_output[max_idx])
            max_idx = i;
    }

    // Infinite loop for debugging
    while (1)
    {
        volatile int8_t keep = debug_output[max_idx];
        (void)keep;
    }
}