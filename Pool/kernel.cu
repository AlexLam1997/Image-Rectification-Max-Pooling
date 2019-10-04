
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include ".\lodepng.h"
#include<algorithm> 
#include<iostream> 

using namespace std;

void process(char* input_filename, char* output_filename);

int main(int argc, char* argv[])
{
    
    cout << max(1,2);
    return 0;
}

void process(char* input_filename, char* output_filename)
{
    unsigned error;
    unsigned char* image, * new_image;
    unsigned width, height;

    error = lodepng_decode32_file(&image, &width, &height, input_filename);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
    new_image = (unsigned char*) malloc(width * height * 4 * sizeof(unsigned char));

    // process image

    for (int i = 0; i < height/2; i+=2) {
        for (int j = 0; j < width/2; j+=2) {

            //value = image[4 * width * i + 4 * j];

            //new_image[4 * width * i + 4 * j + 0] = value; // R
            //new_image[4 * width * i + 4 * j + 1] = value; // G
            //new_image[4 * width * i + 4 * j + 2] = value; // B
            //new_image[4 * width * i + 4 * j + 3] = image[4 * width * i + 4 * j + 3]; // A
        }
    }
  
    
    //unsigned char value;
    //for (int i = 0; i < height; i++) {
    //    for (int j = 0; j < width; j++) {

    //        value = image[4 * width * i + 4 * j];

    //        new_image[4 * width * i + 4 * j + 0] = value; // R
    //        new_image[4 * width * i + 4 * j + 1] = value; // G
    //        new_image[4 * width * i + 4 * j + 2] = value; // B
    //        new_image[4 * width * i + 4 * j + 3] = image[4 * width * i + 4 * j + 3]; // A
    //    }
    //}

    lodepng_encode32_file(output_filename, new_image, width, height);

    free(image);
    free(new_image);
}
