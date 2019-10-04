
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdlib.h>
#include "lodepng.h"

unsigned char rectify(unsigned char input_value) {
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


void pre_thread_process(char* input_filename, char* output_filename, int number_threads) {
	unsigned error;
	char* input_filename, char* output_filename;

	unsigned char* image, * new_image, * cuda_image, * cuda_new_image;
	unsigned width, height;

	error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
	new_image = (unsigned char*)malloc(width * height * 4 * sizeof(unsigned char));

	cudaMalloc(cuda_image, width * height * 4 * sizeof(unsigned char));
	cudaMemcpy(cuda_image, image, width * height * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMalloc(cuda_new_image, width * height * 4 * sizeof(unsigned char));

	threadProcess << < 1, number_threads >> > (height, width, number_threads, cuda_new_image, cuda_image);

	cudaMemcpy(new_image, cuda_new_image, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);//not sure if this is necesary or right???

	lodepng_encode32_file(output_filename, new_image, width, height); //make the new image from the data 

	free(image);
	free(new_image);
	cudaFree(cuda_image);
	cudaFree(cuda_new_image);

}


__global__ threadProcess(int height, int width, int num_threads, unsigned char* new_image, unsigned char* image) {
	// process image
	int start;
	int end;
	int total_size = width * height;
	int thread_size = total_size / num_threads;
	start = thread_size * threadIdx.x;
	end = start + thread_size;

	for (int i = start; i < end; i++) {
		//value = image[4 * width * i + 4 * j];
		new_image[4 * i + 0] = rectify(image[4 * i + 0]); // R
		new_image[4 * i + 1] = rectify(image[4 * i + 1]); // G
		new_image[4 * i + 2] = rectify(image[4 * i + 2]); // B
		new_image[4 * i + 3] = image[4 * i + 3]; // A

	}
}


int main()
{
	///char* input_filename = "C:\Users\jmccon4\source\repos\lab1\lab1\inputFILE.png";
	char* input_filename = "\inputFILE.png";
	char* output_filename = "Output.png";

	pre_thread_process(input_filename, output_filename, 256);

	return 0;
}
