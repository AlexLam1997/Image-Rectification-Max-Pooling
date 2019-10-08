
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdlib.h>
#include "lodepng.h"
#include<time.h>


unsigned char rectify(unsigned char input_value) {
	if (input_value < 127)
	{
		return 127;
	}
	return input_value;
}
__device__ unsigned char rectify_GPU(unsigned char input_value) {
	if (input_value < 127)
	{
		return 127;
	}
	return input_value;

}
void process(char* input_filename, char* output_filename)
{
	unsigned error;
	unsigned char* image, * new_image;
	unsigned width, height;

	error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
	new_image = (unsigned char*)malloc(width * height * 4 * sizeof(unsigned char));

	// process image
	unsigned char value;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			new_image[4 * width * i + 4 * j + 0] = rectify(image[4 * width * i + 4 * j + 0]); // R
			new_image[4 * width * i + 4 * j + 1] = rectify(image[4 * width * i + 4 * j + 1]); // G
			new_image[4 * width * i + 4 * j + 2] = rectify(image[4 * width * i + 4 * j + 2]); // B
			new_image[4 * width * i + 4 * j + 3] = image[4 * width * i + 4 * j + 3]; // A
		}
	}

	lodepng_encode32_file(output_filename, new_image, width, height);

	free(image);
	free(new_image);
}

__global__ void threadProcess(int height, int width, int num_threads, unsigned char* new_image, unsigned char* image, int num_blocks) {
	// process image

	int start;
	int end;
	int total_size = width * height;
	int thread_size = total_size / (num_threads* num_blocks);
	start = thread_size * (blockIdx.x * blockDim.x + threadIdx.x );
	end = start + thread_size;

	for (int i = start; i < end; i++) {


		//value = image[4 * width * i + 4 * j];

		new_image[4 * i + 0] = rectify_GPU(image[4 * i + 0]); // R
		new_image[4 * i + 1] = rectify_GPU(image[4 * i + 1]); // G
		new_image[4 * i + 2] = rectify_GPU(image[4 * i + 2]); // B
		new_image[4 * i + 3] = image[4 * i + 3]; // A

	}


}

void pre_thread_process(char* input_filename, char* output_filename, int number_threads) {
	unsigned error;
	//char* input_filename, char* output_filename;

	unsigned char* image, * new_image, * cuda_image, * cuda_new_image;
	unsigned width, height;

	error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
	new_image = (unsigned char*)malloc(width * height * 4 * sizeof(unsigned char));

	cudaMalloc((void**)&cuda_image,  width * height * 4 * sizeof(unsigned char));
	cudaMemcpy(cuda_image, image, width * height * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&cuda_new_image, width * height * 4 * sizeof(unsigned char));
    
    int block_number = number_threads / 1024 + 1;
    int threads_per_block = number_threads / block_number;

    double time_spent = 0.0;
    clock_t begin = clock();

	threadProcess<<< block_number, threads_per_block >>> (height, width, threads_per_block, cuda_new_image, cuda_image, block_number);

	cudaMemcpy(new_image, cuda_new_image, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);//not sure if this is necesary or right???

	lodepng_encode32_file(output_filename, new_image, width, height); //make the new image from the data 

	free(image);
	free(new_image);
	cudaFree(cuda_image);
	cudaFree(cuda_new_image);

}


int main(int argc, char* argv[])
{
    char* input_filename = argv[1];
    char* output_filename = argv[2];
    //double time_spent = 0.0;
    int thread_nums = atoi(argv[3]);

	//char* input_filename = "\goji.png";
	//char* output_filename = "Output.png";
	////double time_spent = 0.0;
	//int thread_nums[9] = { 1, 2, 4, 8, 16, 32, 64, 128, 256 };

	int i;
	//for (i = 0; i<9; i++) {
		//int number_of_threads = thread_nums[i];
        
		int number_of_threads = thread_nums;
		double time_spent = 0.0;
		clock_t begin = clock();

		pre_thread_process(input_filename, output_filename, number_of_threads);
		clock_t end = clock();
		time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
		printf("Number of threads: %d    Run time %f   \n", number_of_threads, time_spent);
	//}
	return 0;
}
