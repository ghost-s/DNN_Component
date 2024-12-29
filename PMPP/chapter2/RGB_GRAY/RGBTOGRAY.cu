#include <cuda_runtime.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
using namespace std;


__global__ void rgb2gray(unsigned char *rgb, unsigned char *gray, int row, int col) {
    int temp_row = blockIdx.x * blockDim.x + threadIdx.x;
    int temp_col = blockIdx.y * blockDim.y + threadIdx.y;

    if(temp_row < row && temp_col < col){
        int grayoffset = temp_row * col + temp_col; 
        int channel = 3;
        int rgboffset = grayoffset*channel;
        unsigned char r = rgb[rgboffset];
        unsigned char g = rgb[rgboffset + 1];
        unsigned char b = rgb[rgboffset + 2];
        // 使用static_cast将浮点数转化为整数, 显式类型转换更清晰
        gray[grayoffset] = static_cast<unsigned char>(0.21f*r + 0.71f*g + 0.07f*b);
    }
}
int width = 1920;
int height = 1200;
int channels = 3;
unsigned char *rgb;
void read_image_data();
void save_image_data_to_csv(int row, int col, unsigned char* gray);
int main() {
    read_image_data();
    int row = height, col = width;
    int channel = 3;
    unsigned char gray[row * col];
    dim3 threadsperblock(16, 16);
    dim3 blockspergrid(ceil(row / threadsperblock.x), ceil(col / threadsperblock.y));
    unsigned char *rgb_dev, *gray_dev;

    cudaMalloc((void**)&rgb_dev, row * col * channel * sizeof(unsigned char));
    cudaMalloc((void**)&gray_dev, row * col * sizeof(unsigned char));
    cudaMemcpy(rgb_dev, rgb, row * col * channel * sizeof(unsigned char), cudaMemcpyHostToDevice);
    rgb2gray<<<blockspergrid, threadsperblock>>>(rgb_dev, gray_dev, row, col);
    cudaMemcpy(gray, gray_dev, row * col * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(rgb_dev);
    cudaFree(gray_dev);
    save_image_data_to_csv(row, col, gray);
    delete[] rgb;
    return 0;
}

void read_image_data() {
    ifstream file("image_data.csv");

    if (!file.is_open()) {
        cerr << "无法打开文件!" << endl;
        return;
    }
    string line;
    rgb = new unsigned char[width * height * channels];
    // 跳过 CSV 文件的表头
    getline(file, line);

    int index = 0;
    // 读取每一行数据
    while (getline(file, line)) {
        stringstream ss(line);
        string field;
        vector<unsigned char> row_pixels; // 每一行的像素，存储 RGB 值

        while (getline(ss, field, ',')) {
            // 每次读取一个像素的值，并且把它存储进 row_pixels
            rgb[index++] = static_cast<unsigned char>(stoi(field));
        }
    }
    file.close(); // 关闭文件
    
}

void save_image_data_to_csv(int row, int col, unsigned char* gray) {
    ofstream file("gray_image.csv");

    if (!file.is_open()) {
        cerr << "无法打开文件!" << endl;
        return;
    }

    // 将灰度图像数据写入CSV文件
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            file << (int)gray[i * col + j];  // 将数据写入文件并转为整数格式
            if (j < col - 1)
                file << ",";  // 每个元素之间用逗号分隔
        }
        file << endl;  // 每行结束时换行
    }

    file.close();  // 关闭文件
}