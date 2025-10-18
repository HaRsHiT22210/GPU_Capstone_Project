#include "image_utils.h"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <vector>

using namespace cv;

// CUDA kernel for RGB to Grayscale
__global__ void rgbToGrayKernel(unsigned char* d_in, unsigned char* d_out, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        unsigned char r = d_in[idx];
        unsigned char g = d_in[idx + 1];
        unsigned char b = d_in[idx + 2];
        d_out[y * width + x] = 0.299f*r + 0.587f*g + 0.114f*b;
    }
}

bool loadImage(const std::string& filename, Image& img) {
    Mat image = imread(filename, IMREAD_COLOR);
    if (image.empty()) return false;
    img.width = image.cols;
    img.height = image.rows;
    img.channels = image.channels();
    img.data = new unsigned char[img.width * img.height * img.channels];
    std::memcpy(img.data, image.data, img.width * img.height * img.channels);
    return true;
}

bool saveImage(const std::string& filename, const Image& img) {
    Mat output(img.height, img.width, CV_8UC1, img.data);
    return imwrite(filename, output);
}

void convertToGrayGPU(Image& img) {
    unsigned char *d_in, *d_out;
    int img_size = img.width * img.height * img.channels * sizeof(unsigned char);
    int gray_size = img.width * img.height * sizeof(unsigned char);

    cudaMalloc(&d_in, img_size);
    cudaMalloc(&d_out, gray_size);
    cudaMemcpy(d_in, img.data, img_size, cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((img.width+15)/16, (img.height+15)/16);
    rgbToGrayKernel<<<grid, block>>>(d_in, d_out, img.width, img.height, img.channels);
    cudaDeviceSynchronize();

    delete[] img.data;
    img.channels = 1;
    img.data = new unsigned char[gray_size];
    cudaMemcpy(img.data, d_out, gray_size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

std::vector<int> computeHistogramGPU(const Image& img) {
    std::vector<int> hist(256, 0);
    for (int i = 0; i < img.width*img.height; i++) {
        hist[img.data[i]]++;
    }
    return hist;
}

bool saveHistogram(const std::string& filename, const std::vector<int>& hist) {
    std::ofstream file(filename);
    if (!file.is_open()) return false;
    for (int i = 0; i < hist.size(); i++) file << i << "," << hist[i] << "\n";
    return true;
}
