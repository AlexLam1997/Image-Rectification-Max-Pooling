
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include ".\lodepng.h"
#include<algorithm> 
#include<iostream> 

using namespace std;

__global__ void process(char* input_filename, char* output_filename, unsigned width, unsigned height)
{
    // process image

    for (int i = 0; i < height / 2; i += 2) {
        for (int j = 0; j < width / 2; j += 2) {
            

            //new_image[4 * width * i + 4 * j + 0] = value; // R
            //new_image[4 * width * i + 4 * j + 1] = value; // G
            //new_image[4 * width * i + 4 * j + 2] = value; // B
            //new_image[4 * width * i + 4 * j + 3] = image[4 * width * i + 4 * j + 3]; // A
        }
    }
}

int main(int argc, char* argv[])
{
    char* input_filename = argv[1];
    char* output_filename = argv[2];
    int num_threads = atoi(argv[3]);

    unsigned error;
    unsigned char* image, * new_image;
    unsigned char* d_image; 
    unsigned width, height;

    error = lodepng_decode32_file(&image, &width, &height, input_filename);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
    
    //  allocated memory in the device for the input image
    // we dont need it again the the host so just do cudaMalloc
    size_t imageSize = (size_t) width * height * 4 * sizeof(unsigned char);
    cudaMalloc((void** ) & d_image, imageSize);
    cudaMemcpy(d_image, image, imageSize, cudaMemcpyHostToDevice);
    // allocate shared memory for the new image because we want it in host
    cudaMallocManaged(&new_image, imageSize/4);

    process<<<1, num_threads>>>(d_image, new_image, width, height);

    cudaDeviceSynchronize();

    lodepng_encode32_file(output_filename, new_image, width, height);

    cudaFree(d_image); cudaFree(new_image);
    return 0;
}
