
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include ".\lodepng.h"
#include <algorithm> 

using namespace std;

__device__ int max(int a, int b, int c, int d) {
	int max = a;
	if (b > max) max = b;
	if (c > max) max = c;
	if (d > max) max = d;
	return max;
}

__global__ void process(unsigned char* input_image, unsigned char* output_image, unsigned width, unsigned height, int num_threads)
{
	int start;
	int end;
	// Number of pixels to process
	int total_size = width * height;
	// Number of pixels per thread
	int thread_size = total_size / num_threads;
	int blocks_per_thread = thread_size / 4;
	start = blocks_per_thread * threadIdx.x;
	end = start + blocks_per_thread;
	 
    // process image
	// split image into N 2x2 blocks
	// each thread processes N/numThreads blocks 
	// first square: tid * 8 (tid = 0) 
	// below first: tid*8 + width * 8
    for (int i = start; i< end; i++){
		unsigned char* one = input_image + 4 * i;
		unsigned char* two = input_image+ 4 * i + 4;
		unsigned char* three = input_image + 4 * i + 4 * width;
		unsigned char* four = input_image + 4 * i + 4 * width + 4;

		int maxR = max( *one, *two, *three, *four );
		int maxG = max( *(one+1), *(two+1), *(three+1), *(four+1));
		int maxB = max( *(one + 2), *(two + 2), *(three + 2), *(four + 2) );
		int maxA = max( *(one + 3), *(two + 3), *(three + 3), *(four + 3) );

		output_image[4 * i] = maxR;
		output_image[4 * i + 1] = maxG;
		output_image[4 * i + 2] = maxB;
		output_image[4 * i + 3] = maxA;
    }
}

int main(int argc, char* argv[])
{
	char* input_filename = "input800x923.png";//argv[1];
	char* output_filename = "output.png";//argv[2];
	int num_threads = 10;//atoi(argv[3]);

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

    process<<<1, num_threads>>>(d_image, new_image, width, height, num_threads);

    cudaDeviceSynchronize();

    lodepng_encode32_file(output_filename, new_image, width, height);

    cudaFree(d_image); cudaFree(new_image);
    return 0;
}
