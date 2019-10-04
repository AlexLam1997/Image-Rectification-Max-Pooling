
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdlib.h>
#include "lodepng.h"

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

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

__global__ void threadProcess(int height, int width, int num_threads, unsigned char* new_image, unsigned char* image) {
	// process image

	int start;
	int end;
	int total_size = width * height;
	int thread_size = total_size / num_threads;
	start = thread_size * threadIdx.x;
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

	threadProcess<<< 1, number_threads >> > (height, width, number_threads, cuda_new_image, cuda_image);

	cudaMemcpy(new_image, cuda_new_image, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);//not sure if this is necesary or right???

	lodepng_encode32_file(output_filename, new_image, width, height); //make the new image from the data 

	free(image);
	free(new_image);
	cudaFree(cuda_image);
	cudaFree(cuda_new_image);

}





int main()
{
	///char* input_filename = "C:\Users\jmccon4\source\repos\lab1\lab1\inputFILE.png";
	char* input_filename = "\inputFILE.png";
	char* output_filename = "Output.png";

	pre_thread_process(input_filename, output_filename, 256);

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)& dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel <<<1, size >> > (dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
