#include <iostream>
#include "image_utils.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: ./gpu_image_proc <input_image> <output_image>\n";
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];

    std::cout << "Loading image: " << input_file << std::endl;
    Image img;
    if (!loadImage(input_file, img)) {
        std::cerr << "Failed to load image.\n";
        return 1;
    }

    std::cout << "Converting to grayscale using GPU...\n";
    convertToGrayGPU(img);

    std::cout << "Saving output image: " << output_file << std::endl;
    if (!saveImage(output_file, img)) {
        std::cerr << "Failed to save image.\n";
        return 1;
    }

    std::cout << "Computing histogram...\n";
    auto hist = computeHistogramGPU(img);
    saveHistogram("data/output/histogram.csv", hist);

    std::cout << "Done!\n";
    return 0;
}
