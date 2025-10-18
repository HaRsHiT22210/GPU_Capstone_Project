#pragma once
#include <string>
#include <vector>

struct Image {
    int width;
    int height;
    int channels;
    unsigned char* data;
};

bool loadImage(const std::string& filename, Image& img);
bool saveImage(const std::string& filename, const Image& img);

void convertToGrayGPU(Image& img);
std::vector<int> computeHistogramGPU(const Image& img);
bool saveHistogram(const std::string& filename, const std::vector<int>& hist);
