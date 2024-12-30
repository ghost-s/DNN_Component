#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
using namespace std;

__global__ void blurkernel(unsigned char *rgb, unsigned char *blur, int row, int col, int blursize){
    int temp_row = blockIdx.x * blockDim.x + threadIdx.x;
    int temp_col = blockIdx.y * blockDim.y + threadIdx.y;
    if(temp_row < row && temp_col < col){
        int channel = 3;
        int numPixel = 0;
        int rPixel = 0;
        int gPixel = 0;
        int bPixel = 0;
        for(int i = -blursize; i <= blursize; i++){
            for(int j = -blursize; j <= blursize; j++){
                int currow = temp_row + i;
                int curcol = temp_col + j;
                if(currow >= 0 && curcol >= 0 && currow < row && curcol < col){
                    int rgboffset = (currow * col + curcol) * channel;
                    numPixel++;
                    rPixel += rgb[rgboffset];
                    gPixel += rgb[rgboffset + 1];
                    bPixel += rgb[rgboffset + 2];
                }
            }
        }
        int bluroffset = (temp_row * col + temp_col) * channel;
        if(numPixel > 0){
            blur[bluroffset] = static_cast<unsigned char>(rPixel / numPixel);
            blur[bluroffset + 1] = static_cast<unsigned char>(gPixel / numPixel);
            blur[bluroffset + 2] = static_cast<unsigned char>(bPixel / numPixel);
        }else{
            blur[bluroffset] = rgb[bluroffset];
            blur[bluroffset + 1] = rgb[bluroffset + 1];
            blur[bluroffset + 2] = rgb[bluroffset + 2];
        }
    }
}

int width = 437;
int height = 332;
int channels = 3;
unsigned char *rgb;
void read_image_data();
void save_to_csv(unsigned char* blur, int row, int col);

int main() {
    read_image_data();
    int row = height, col = width;
    int channel = 3;
    int blursize = 2;
    unsigned char gray[row * col * channel];
    dim3 threadsperblock(16, 16);
    dim3 blockspergrid(ceil(row / threadsperblock.x), ceil(col / threadsperblock.y));
    unsigned char *rgb_dev, *gray_dev;

    cudaMalloc((void**)&rgb_dev, row * col * channel * sizeof(unsigned char));
    cudaMalloc((void**)&gray_dev, row * col * channel * sizeof(unsigned char));
    cudaMemcpy(rgb_dev, rgb, row * col * channel * sizeof(unsigned char), cudaMemcpyHostToDevice);

    blurkernel<<<blockspergrid, threadsperblock>>>(rgb_dev, gray_dev, row, col, blursize);

    cudaMemcpy(gray, gray_dev, row * col * channel * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(rgb_dev);
    cudaFree(gray_dev);
    save_to_csv(gray, row, col);
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

void save_to_csv(unsigned char* blur, int row, int col) {
    std::ofstream file("blurred_image.csv");

    if (!file.is_open()) {
        std::cerr << "无法打开文件!" << std::endl;
        return;
    }

    // 写入 CSV 文件的表头
    file << "R,G,B" << std::endl;

    // 遍历每个像素并将 RGB 值写入文件
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            int offset = (i * col + j) * 3;  // 计算该像素的RGB偏移量
            file << (int)blur[offset] << ","  // R 值
                 << (int)blur[offset + 1] << ","  // G 值
                 << (int)blur[offset + 2];  // B 值
            file << std::endl;  // 每个像素的 RGB 写入后换行
        }
    }

    file.close();  // 关闭文件
}