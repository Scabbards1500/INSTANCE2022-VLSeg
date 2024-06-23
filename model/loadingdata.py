import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class INSTANCE2022(Dataset):

    def __init__(self, image, mask, text, tokenizer):

        super(INSTANCE2022, self).__init__()

        # 相当于这个地方要通过一个csv文件把数据集的路径读进来,但是我寻思我们输入的时候已经是正序了
        # with open(csv_path, 'r') as f:
        #     self.data = pd.read_csv(f)
        # self.image_list = list(self.data['Image'])
        # self.caption_list = list(self.data['Description'])

        self.images = image
        self.masks = mask
        self.text = text

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):

        return len(self.text['data'])

    def __getitem__(self, idx):

        image = self.images[idx]
        label = self.masks[idx]
        caption = self.text['description'][idx]
        #这里图像都已经处理好了，直接输入就行

        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=24,
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token,mask = token_output['input_ids'],token_output['attention_mask']

        data = {'image':image, 'label':label, 'token':token, 'mask':mask}

        image,label,token,mask = data['image'],data['gt'],data['token'],data['mask']
        text = {'input_ids':token.squeeze(dim=0), 'attention_mask':mask.squeeze(dim=0)}
        print("2aasdasdasdasd")
        print(text)

        return ([image, text], label)

# if __name__ == '__main__':
#     print("hello")






class datasetModelClassifywithopencv(Dataset):
    def __init__(self, images, labels, text, tokenizer, targetsize=(1, 512, 512)):
        super(datasetModelClassifywithopencv).__init__()

        self.labels = labels
        self.images = images
        self.targetsize = targetsize
        self.text = text
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        image should normalization,and npy type
        :param index:
        :return:
        """
        imagepath = self.images[index]
        # load image
        image = cv2.imread(imagepath, 0)
        # resize image to fixed size
        image = cv2.resize(image, (self.targetsize[1], self.targetsize[2]))
        # normalization image to zscore
        image = (image - image.mean()) / image.std()
        # transpose (H,W,C) order to (C,H,W) order
        H, W = np.shape(image)[0], np.shape(image)[1]
        image = np.reshape(image, (H, W, 1))
        image = np.transpose(image, (2, 0, 1))
        assert image.shape[0] == self.targetsize[0] and image.shape[1] == self.targetsize[1] and image.shape[2] == \
               self.targetsize[2]
        # convert numpy to tensor
        images_tensor = torch.as_tensor(image).float()  # transform ndarray to tensor
        #text
        caption = self.text['description'][index]
        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=24,
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token, mask = token_output['input_ids'], token_output['attention_mask']


        # torch.set_printoptions(8)
        label = self.labels[index]
        label = int(label)
        label_tensor = torch.as_tensor(label).long()

        data = {'image': image, 'label': label, 'token': token, 'mask': mask}
        image, label, token, mask = data['image'], data['gt'], data['token'], data['mask']
        text = {'input_ids': token.squeeze(dim=0), 'attention_mask': mask.squeeze(dim=0)}

        return {'train_data': [images_tensor, text], 'label': label_tensor}