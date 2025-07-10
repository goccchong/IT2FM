from PIL import Image
import os

def tif_to_png(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            # 构建输入和输出文件的完整路径
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename[:-4] + '.png')

            # 打开.tif文件并保存为.png格式
            try:
                img = Image.open(input_path)
                img.save(output_path, 'PNG')
                print(f"Converted {filename} to PNG format.")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

def delete_tif_files(folder_path):
    # 遍历文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.tif'):
                # 构建要删除的文件的完整路径
                file_path = os.path.join(root, file)
                # 删除文件
                try:
                    os.remove(file_path)
                    print(f"Deleted {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

# 指定输入和输出文件夹
input_folder = '/home/ps/data/code/wgy/work_03/SFFNet-main/results/potsdam/our5/gray/'
output_folder = '/home/ps/data/code/wgy/work_03/SFFNet-main/results/potsdam/our5/gray/'

# 执行转换
#tif_to_png(input_folder, output_folder)

delete_tif_files(output_folder)
