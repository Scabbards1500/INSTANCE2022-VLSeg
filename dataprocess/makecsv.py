import os
import csv


data_dir_train = r'D:/python/INSTANCE2022-INtracranial-hemorrhage-SegmenTAtioN-ChallengE-main/dataset/train/data'
label_dir_train = r'D:/python/INSTANCE2022-INtracranial-hemorrhage-SegmenTAtioN-ChallengE-main/dataset/train/label'
output_csv_train = r'D:\python\INSTANCE2022-INtracranial-hemorrhage-SegmenTAtioN-ChallengE-main\dataprocess\data\train.csv'

data_dir_val = r'D:/python/INSTANCE2022-INtracranial-hemorrhage-SegmenTAtioN-ChallengE-main/dataset/val/data'
label_dir_val = r'D:/python/INSTANCE2022-INtracranial-hemorrhage-SegmenTAtioN-ChallengE-main/dataset/val/label'
output_csv_val = r'D:\python\INSTANCE2022-INtracranial-hemorrhage-SegmenTAtioN-ChallengE-main\dataprocess\data\validation.csv'

data_dir_aug = r'D:/python/INSTANCE2022-INtracranial-hemorrhage-SegmenTAtioN-ChallengE-main/dataset/augtrain/data'
label_dir_aug = r'D:/python/INSTANCE2022-INtracranial-hemorrhage-SegmenTAtioN-ChallengE-main/dataset/augtrain/label'
output_csv_aug = r'D:\python\INSTANCE2022-INtracranial-hemorrhage-SegmenTAtioN-ChallengE-main\dataprocess\data\trainaugdata.csv'



def generate_csv(data_dir, label_dir, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['data', 'label'])

        data_files = os.listdir(data_dir)
        for index, data_file in enumerate(data_files):
            data_path = os.path.join(data_dir, data_file).replace('/', '\\')
            label_path = os.path.join(label_dir, data_file).replace('/', '\\')
            print(data_path)
            print(label_path)
            csv_writer.writerow([data_path, label_path])

# generate_csv(data_dir_train, label_dir_train, output_csv_train)
# generate_csv(data_dir_val, label_dir_val, output_csv_val)
generate_csv(data_dir_aug, label_dir_aug, output_csv_aug)
