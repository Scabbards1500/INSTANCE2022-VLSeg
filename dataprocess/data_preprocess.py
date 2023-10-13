import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nibabel as nib

# 设置路径
data_dir = r'D:\tempdataset\train'
image_dir = os.path.join(data_dir, 'data')
label_dir = os.path.join(data_dir, 'label')
output_dir = r'D:\python\INSTANCE2022-INtracranial-hemorrhage-SegmenTAtioN-ChallengE-main\dataset'

# 获取所有文件名
image_files = sorted(os.listdir(image_dir))
label_files = sorted(os.listdir(label_dir))

# 划分数据集
train_image_files, val_image_files, train_label_files, val_label_files = train_test_split(
    image_files, label_files, test_size=0.2, random_state=42
)

# print(len(train_image_files), len(train_label_files))
# print(len(val_image_files), len(val_label_files))

# 创建输出文件夹
os.makedirs(os.path.join(output_dir, 'train', 'data'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train', 'label'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val', 'data'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val', 'label'), exist_ok=True)

# 生成CSV数据 for Train
train_data = []
for idx, (image_file, label_file) in enumerate(zip(train_image_files, train_label_files)):
    image_path = os.path.join(output_dir, 'train', 'data', f'{idx}.npy')
    label_path = os.path.join(output_dir, 'train', 'label', f'{idx}.npy')

    # 将NIfTI文件转换为Numpy数组，并保存为.npy
    image_data = nib.load(os.path.join(image_dir, image_file)).get_fdata()
    label_data = nib.load(os.path.join(label_dir, label_file)).get_fdata()

    np.save(image_path, image_data)
    np.save(label_path, label_data)

    train_data.append([image_path, label_path])

# 生成CSV数据 for Validation
val_data = []
for idx, (image_file, label_file) in enumerate(zip(val_image_files, val_label_files)):
    image_path = os.path.join(output_dir, 'val', 'data', f'{idx}.npy')
    label_path = os.path.join(output_dir, 'val', 'label', f'{idx}.npy')

    # 将NIfTI文件转换为Numpy数组，并保存为.npy
    image_data = nib.load(os.path.join(image_dir, image_file)).get_fdata()
    label_data = nib.load(os.path.join(label_dir, label_file)).get_fdata()

    np.save(image_path, image_data)
    np.save(label_path, label_data)

    val_data.append([image_path, label_path])

# 创建CSV文件 for Train
train_csv_data = pd.DataFrame(train_data, columns=['data', 'label'])
train_csv_data.to_csv(os.path.join(output_dir, 'train', 'traindata.csv'), index=False)

# 创建CSV文件 for Validation
val_csv_data = pd.DataFrame(val_data, columns=['data', 'label'])
val_csv_data.to_csv(os.path.join(output_dir, 'val', 'validata.csv'), index=False)

print("数据集划分和CSV文件生成完成！")
