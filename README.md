## Reproduce the code of "TieFake: Title-Text Similarity and Emotion-Aware Fake News Detection"
# Start

To run the code in this repo, you need to have `Python>=3.9.6`, `PyTorch>=1.9.0`

Other dependencies can be installed using the following commands:
`pip install -r requirements.txt`

The organization of the data folder is as follows：
```
--Data
  --gossipcop_images
    --xx.jpg
    ......
  --processed_data
    --gossipcop_v3-1_style_based_fake.tsv
    --gossipcop_v3-4_story_based_fake.tsv
  --raw_data
  --target_data
    --gossipcop_v3-1_style_based_fake.json
    --gossipcop_v3-4_story_based_fake.json
  --v3-1
    --gossipcop_train.tsv
    --gossipcop_test.tsv
  ......
```

# Running
1. run `Data/dataprocess.py` to merge each dataset and split train and test dataset, then the processed dataset will be placed in `Data/processed_data` folder, train and test dataset will be placed in `Data/{dataset_floder}/` folder, dataset_floder: `['v3-1', '3-1_generated', 'v3-4', '3-4_generated', '3-4*']`
2. run `bert_training.py` to train bert in these datasets
3. run `resnest101_training.py` to train resnest_101 in these datasets
4. run `main.py` to train fusion_model

**Notice：** 
1. Modify the following code in `main.py` to specify the dataset you want to train.
```
folder = "v3-1" # 当前训练的数据集 ['v3-1', '3-1_generated', 'v3-4', '3-4_generated', '3-4*']
generated = False # 是否使用LLM生成的新数据集 ['True', 'False']
```
2. Modify the `text_sent = self.csv_frame.loc[idx, 'text']` in `dataloader.py` to change news texts you will train. If you want to train original texts, then use `text_sent = self.csv_frame.loc[idx, 'origin_text']`, if you want to train generated texts, then use `text_sent = self.csv_frame.loc[idx, 'generated_text']`, The `dataloader.py` contains three relevant lines of code, please make sure to modify all of them.

# Result 
The experimental results of the replication are presented in the table below.

| Dataset         | **ACC-T** | **ACC-F** | ACC   | PRE   | REC   | F1    |
|-----------------|-------|-------|-------|-------|-------|-------|
| Original Result | -     | -     | 0.8920| 0.8870| 0.9020| 0.8940|
| v3-1 origin_text| 0.6460| 0.8948| 0.8395| 0.6367| 0.6460| 0.8967|
| v3-1 generated_text| 0.7956| 0.9281| 0.8987| 0.7596| 0.7956| 0.9345|
| v3-4 origin_text| 0.6357| 0.8805| 0.8263| 0.6021| 0.6357| 0.8876|
| v3-4 generated_text| 0.6059| 0.8795| 0.8189| 0.5884| 0.6059| 0.8832|
| v3-4* generated_text| 0.9044| 0.5565| 0.7332| 0.6780| 0.9044| 0.6724|



