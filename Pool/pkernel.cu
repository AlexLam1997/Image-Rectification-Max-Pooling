
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include ".\lodepng.h"
#include <algorithm> 
#include<time.h>
using namespace std;


__device__ int max(int a, int b, int c, int d) {
	int max = a;
	if (b > max) max = b;
	if (c > max) max = c;
	if (d > max) max = d;
	return max;
}

__global__ void gpu_process(unsigned char* input_image, unsigned char* output_image, unsigned width, unsigned height, int num_threads, int num_blocks)
{
	int start;
	int end;

	// Number of pixels to process
	int total_size = width * height;

	// Number of pixels per thread
	int thread_size = total_size / (num_threads* num_blocks);
	int squares_per_thread = thread_size / 4;
    int squares_per_row = width/2; 

	start = squares_per_thread * (blockIdx.x * blockDim.x + threadIdx.x);
	end = start + squares_per_thread;
	 
    // process image
	// split image into N 2x2 squares
	// each thread processes N/numThreads squares 
	// first square: tid * 8 (tid = 0) 
	// below first: tid*8 + width * 8
    // i is block number
    for (int i = start; i< end; i++){
        int row = 2*(i/squares_per_row);
        unsigned char* one = input_image + i % squares_per_row * 4 * 2 + row * width*4;
        unsigned char* two = one + 4;
        unsigned char* three = one+ width*4;
        unsigned char* four = two + width*4;

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

double run_process(int num_threads, int width, int height, unsigned char* new_image, char* output_filename, unsigned char* d_image) {
	int block_number = num_threads / 1024 + 1;
	int threads_per_block = num_threads / block_number;

	double time_spent = 0.0;
	clock_t begin = clock();
	gpu_process <<<block_number, threads_per_block >>> (d_image, new_image, width, height, threads_per_block, block_number);
	cudaDeviceSynchronize();
	lodepng_encode32_file(output_filename, new_image, width / 2, height / 2);
	clock_t end = clock();

	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
	return time_spent;
}

void pre_process(char* input_filename, unsigned char** d_image, unsigned char** new_image, unsigned* width, unsigned* height) {
	unsigned error;
	unsigned char* image;

	error = lodepng_decode32_file(&image, width, height, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

	// allocated memory in the device for the input image
	// we dont need it again the the host so just do cudaMalloc
	size_t imageSize = (size_t)((*width) * (*height) * 4 * sizeof(unsigned char));
	cudaMalloc((void**) & *d_image, imageSize);
	cudaMemcpy(*d_image, image, imageSize, cudaMemcpyHostToDevice);
	// allocate shared memory for the new image because we want it in host
	cudaMallocManaged(new_image, imageSize / 4);
}

int main(int argc, char* argv[])
{
	bool use_cli = false; 
    char* input_filename = argv[1];
    char* output_filename = argv[2];

	unsigned char * new_image;
	unsigned char* d_image;
	unsigned width, height;

	pre_process(input_filename, &d_image, &new_image, &width, &height);

	if (use_cli) {
		// use number of threads provided by command line, use this for demos
		int number_of_threads = atoi(argv[3]);
		double duration = run_process(number_of_threads, width, height, new_image, output_filename, d_image);
		printf("Number of threads: %d    Run time %f   \n", number_of_threads, duration);
	}
	else {
		// disreguard command line thread numbers and use preset number of threads, use this for timing information gathering
		int max_thread_power = 11;
		int average_count = 100;
		double average_time = 0;
		for (int i = 0; i <= max_thread_power; i++) {
			int number_of_threads = pow(2, i);
			for (int j = 0; j < average_count; j++) {
				double duration = run_process(number_of_threads, width, height, new_image, output_filename, d_image);
				average_time += duration/average_count;
			}
			printf("Average runtime for %d threads and %d runs is:     %f seconds.\n", number_of_threads, average_count, average_time);
		}
	}

    cudaFree(d_image); cudaFree(new_image);
    return 0;
}
