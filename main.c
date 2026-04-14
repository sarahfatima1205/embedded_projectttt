#include "stm32f4xx_hal.h"
#include "model.h"

// Debug variables
volatile int8_t debug_input[INPUT_SIZE];
volatile int8_t debug_output[OUTPUT_SIZE];

int main(void)
{
    HAL_Init();

    // Example input (replace later with real MNIST sample)
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        debug_input[i] = (i % 2 == 0) ? 10 : -10;
    }

    run_inference((int8_t*)debug_input, (int8_t*)debug_output);

    // Find predicted digit
    int max_idx = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++)
    {
        if (debug_output[i] > debug_output[max_idx])
            max_idx = i;
    }

    // Infinite loop for debugging
    while (1)
    {
        // Check:
        // debug_output[]
        // max_idx
    }
}