# import torch
# import numpy as np
# from torchvision.io import read_image
# import csv

# # 读取图像
# x = read_image("zly.jpg").contiguous().cuda()
# print("Input image:", x.shape, x.dtype)
# # 确保 dtype 为 uint8
# assert x.dtype == torch.uint8
# print("Input image:", x.shape, x.dtype)

# # 将张量从 GPU 转移到 CPU，因为 CSV 文件需要 CPU 上的数据
# x_cpu = x.cpu()

# # 将张量的形状从 (3, 1200, 1920) 转换为 (3, 1200 * 1920)
# # 然后将其重塑为一个包含每个像素数据的列表
# image_data = x_cpu.permute(1, 2, 0).reshape(-1, 3).numpy()

# # 将数据保存到 CSV 文件
# csv_filename = "image_data.csv"
# with open(csv_filename, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["R", "G", "B"])  # 写入列名
#     writer.writerows(image_data)  # 写入每个像素的 RGB 值
import torch
import numpy as np
import matplotlib.pyplot as plt

# 读取 CSV 文件并转换为 PyTorch 张量
def read_image_from_csv(file_path, row, col):
    # 读取 CSV 文件
    data = np.loadtxt(file_path, delimiter=',', dtype=np.uint8)

    # 检查数据形状是否正确
    if data.shape != (row, col):
        raise ValueError(f"读取的数据形状不匹配，期望形状: ({row}, {col}), 实际形状: {data.shape}")

    # 转换为 PyTorch 张量
    tensor = torch.tensor(data, dtype=torch.uint8)

    return tensor

# 保存张量为图像
def save_image(tensor, output_path):
    # 将张量转换为 numpy 数组，并确保数据类型为 uint8
    image = tensor.numpy().astype(np.uint8)

    # 使用 matplotlib 显示图像
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # 不显示坐标轴
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

# 主函数
def main():
    row = 1200
    col = 1920

    # 从 CSV 文件读取数据并转换为张量
    tensor = read_image_from_csv('gray_image.csv', row, col)

    # 输出图像
    save_image(tensor, 'output_image.png')

    print("图像已保存为 output_image.png")

if __name__ == "__main__":
    main()
