from __future__ import print_function, division

import os
import SimpleITK as sitk
import numpy as np
from dataprocess.utils import ConvertitkTrunctedValue, resize_image_itkwithsize
from dataprocess.utils import file_name_path, GetLargestConnectedCompont, GetLargestConnectedCompontBoundingbox, \
    MorphologicalOperation

image_dir = "data"
mask_dir = "label"
image_pre = ".nii.gz"
mask_pre = ".nii.gz"


def preparesampling3dtraindata(datapath, trainImage, trainMask, shape=(96, 96, 96)):
    newSize = shape
    dataImagepath = datapath + "\\" + image_dir
    dataMaskpath = datapath + "\\" + mask_dir
    print("hello")
    print(dataImagepath)
    all_files = file_name_path(dataImagepath, False, True)
    for image_name in all_files:  # Iterate over the original file names
        mask_gt_file = dataMaskpath + "/" + image_name
        masksegsitk = sitk.ReadImage(mask_gt_file, sitk.sitkUInt8)
        image_gt_file = dataImagepath + "/" + image_name
        imagesitk = sitk.ReadImage(image_gt_file, sitk.sitkInt16)

        _, resizeimage = resize_image_itkwithsize(imagesitk, newSize, imagesitk.GetSize(), sitk.sitkLinear)
        resizeimage = ConvertitkTrunctedValue(resizeimage, 100, 0, 'meanstd')
        resizemaskarray, resizemask = resize_image_itkwithsize(masksegsitk, newSize, masksegsitk.GetSize(),
                                                               sitk.sitkNearestNeighbor)
        resizeimagearray = sitk.GetArrayFromImage(resizeimage)
        # step 3 get subimages and submasks
        if not os.path.exists(trainImage):
            os.makedirs(trainImage)
        if not os.path.exists(trainMask):
            os.makedirs(trainMask)
        # Extract the original file name without extension
        file_base_name = os.path.splitext(image_name)[0]
        filepath1 = os.path.join(trainImage, file_base_name[0:3] + ".npy")
        filepath = os.path.join(trainMask, file_base_name[0:3] + ".npy")
        np.save(filepath1, resizeimagearray)
        np.save(filepath, resizemaskarray)


def preparetraindata():
    """
    :return:
    """
    src_train_path = r"D:\tempdataset\50data_origen\train"
    source_process_path = r"D:\tempdataset\50data\train"
    outputimagepath = source_process_path + "/" + image_dir
    outputlabelpath = source_process_path + "/" + mask_dir
    preparesampling3dtraindata(src_train_path, outputimagepath, outputlabelpath, (256, 320, 32))


def preparevalidationdata():
    """
    :return:
    """
    src_train_path = r"D:\tempdataset\50data_origen\val"
    source_process_path = r"D:\tempdataset\50data\val"
    outputimagepath = source_process_path + "/" + image_dir
    outputlabelpath = source_process_path + "/" + mask_dir
    preparesampling3dtraindata(src_train_path, outputimagepath, outputlabelpath, (256, 320, 32))


if __name__ == "__main__":
    preparetraindata()
    # preparevalidationdata()
