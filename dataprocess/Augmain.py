from dataprocess.Augmentation.ImageAugmentation import DataAug3D

if __name__ == '__main__':
    aug = DataAug3D(rotation=10, width_shift=0.01, height_shift=0.01, depth_shift=0, zoom_range=0,
                    vertical_flip=True, horizontal_flip=True)
    aug.DataAugmentation('D:\python\INSTANCE2022-INtracranial-hemorrhage-SegmenTAtioN-ChallengE-main\dataprocess\data\\traindata.csv', 10, aug_path=r'D:\python\INSTANCE2022-INtracranial-hemorrhage-SegmenTAtioN-ChallengE-main\dataset\augtrain')
