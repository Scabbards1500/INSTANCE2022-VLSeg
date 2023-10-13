import pandas as pd

# 读取 Train_ID.csv 和 Val_ID.csv 中的索引
train_index_df = pd.read_csv(r'D:\python\INSTANCE2022-INtracranial-hemorrhage-SegmenTAtioN-ChallengE-main\dataprocess\data\traindata.csv')
val_index_df = pd.read_csv(r'D:\python\INSTANCE2022-INtracranial-hemorrhage-SegmenTAtioN-ChallengE-main\dataprocess\data\validata.csv')

# 读取 Train_Val_test.csv 中的数据
data_df = pd.read_csv(r'D:\python\INSTANCE2022-INtracranial-hemorrhage-SegmenTAtioN-ChallengE-main\dataprocess\data\train_valdata.csv')

print(train_index_df['data'].str[-7:])
# 根据索引划分数据集
train_data = data_df[data_df['data'].isin(train_index_df['data'].str[-7:])]
val_data = data_df[data_df['data'].isin(val_index_df['data'].str[-7:])]

# 将划分后的数据保存到 Train_text.xlsx 和 Val_text.xlsx
train_data.to_csv('Train_text.csv', index=False) # index=False 不保存索引
val_data.to_csv('Val_text.csv', index=False)

print("Data split and saved successfully.")
