import pandas as pd
import json
from tqdm import tqdm

# 合并数据集
def merge_dataset(sub_data_name):
    # 读取gossipcop_all.tsv文件
    gossipcop_all_path = './Data/gossipcop_all.tsv'
    df_all = pd.read_csv(gossipcop_all_path, sep='\t')
    sub_data_path = './Data/target_dataset/' + sub_data_name + '.json'

    # 初始化空列表，用于存储匹配成功的数据
    data = []

    # 遍历gossipcop_all.tsv数据集
    for index, row in tqdm(df_all.iterrows()):
        news_id = row['id']
        title = row['title']
        label = 0
        # sub_data_path: 目标数据集地址  读取gossipcop_v3-1_style_based_fake.json文件
        with open(sub_data_path, 'r') as f_fake:
            fake_data = json.load(f_fake)
            for key, value in fake_data.items():
                if value['origin_id'] == news_id and value['has_top_img'] == 1:
                    # if value['origin_label'] == "fake":
                    #     label = 0
                    # else:
                    #     label = 1
                    # data.append({
                    #     'id': news_id,
                    #     "title": title,
                    #     'origin_text': value['origin_text'].replace('\n', '.'),
                    #     'generated_text': value['generated_text'].replace('\n', '.'),
                    #     'label': label  # 0代表假新闻,1代表真新闻
                    # })
                    data.append({
                        'id': news_id,
                        "title": title,
                        'text': value['generated_text'].replace('\n', '.'),
                        # 'generated_text': value['generated_text_glm4'].replace('\n', '.'),
                        'label': 0  # 0代表假新闻,1代表真新闻
                    })
                    data.append({
                        'id': news_id,
                        "title": title,
                        'text': value['origin_text'].replace('\n', '.'),
                        # 'origin_text': value['origin_text'].replace('\n', '.'),
                        'label': 1  # 0代表假新闻,1代表真新闻
                    })
        
    # 将匹配成功的数据转换为DataFrame
    df_result = pd.DataFrame(data)
    # 将结果保存为新的tsv文件
    df_result.to_csv(f'./Data/processed_data/*_{sub_data_name}.tsv', sep='\t', index=False)

# sub_data_name = "gossipcop_v3-4_story_based_fake"
# merge_dataset(sub_data_name)


# 划分数据集
def split_dataset(data_path, folder):
    # 读取数据集(总tsv文件)
    df = pd.read_csv(data_path, sep='\t')

    # 打乱数据集
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 计算划分的数量
    total_samples = len(df)
    train_samples = int(total_samples * 0.8)
    val_samples = int(total_samples * 0.1)

    # 划分数据集
    df_train = df[:train_samples]
    df_val = df[train_samples:train_samples+val_samples]
    df_test = df[train_samples+val_samples:]

    # 将划分后的数据集保存为新的文件
    df_train.to_csv(f'./Data/{folder}/gossipcop_train.tsv', sep='\t', index=False)
    df_val.to_csv(f'./Data/{folder}/gossipcop_val.tsv', sep='\t', index=False)
    df_test.to_csv(f'./Data/{folder}/gossipcop_test.tsv', sep='\t', index=False)


data_path = './Data/processed_data/*_gossipcop_v3-4_story_based_fake.tsv'
folder = 'v3-4*'
split_dataset(data_path, folder)

# # 遍历gossipcop_all.tsv数据集
# for index, row in tqdm(df_all.iterrows()):
#     news_id = row['id']
#     title = row['title']
#     label = 0
#     # 读取gossipcop_v3-2_content_based_fake.json文件
#     with open('./Data/target_dataset/gossipcop_v3-2_content_based_fake.json', 'r') as f_fake:
#         fake_data = json.load(f_fake)
#         for key, value in fake_data.items():
#             if value['origin_id'] == news_id and value['has_top_img'] == 1:
#                 if value['origin_label'] == "fake":
#                     label = 0
#                 else:
#                     label = 1
#                 data.append({
#                     'id': news_id,
#                     "title": title,
#                     'text': value['generated_text_glm4'].replace('\n', '.'),
#                     # 'generated_text': value['generated_text_glm4'].replace('\n', '.'),
#                     'label': 0  # 0代表假新闻,1代表真新闻
#                 })
#                 data.append({
#                     'id': news_id,
#                     "title": title,
#                     'text': value['origin_text'].replace('\n', '.'),
#                     # 'origin_text': value['origin_text'].replace('\n', '.'),
#                     'label': 1  # 0代表假新闻,1代表真新闻
#                 })



