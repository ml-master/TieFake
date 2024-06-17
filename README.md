## TieFake: Title-Text Similarity and Emotion-Aware Fake News Detection (TieFake).
# Start

To run the code in this repo, you need to have `Python>=3.9.6`, `PyTorch>=1.9.0`
Other dependencies can be installed using the following commands:

use `pip install -r requirements.txt` to download datasets

The organization of the data folder is as follows：
```
--Data
  --gossipcop_images
    --xx.jpg
    ......
  --processed_data
    --gossipcop_v3-1_style_based_fake.tsv
    --gossipcop_v3-2_content_based_fake.tsv
    --gossipcop_v3-3_integration_based_fake_tn200.tsv
    --gossipcop_v3-4_story_based_fake.tsv
  --raw_data
  --target_data
    --gossipcop_v3-1_style_based_fake.json
    --gossipcop_v3-2_content_based_fake.json
    --gossipcop_v3-3_integration_based_fake_tn200.json
    --gossipcop_v3-4_story_based_fake.json
  --v3-1
    --gossipcop_train.tsv
    --gossipcop_test.tsv
  ......
```

run Data/dataprocess.py to merge each dataset and split train and test dataset
run bert_training.py to train bert in these datasets
run resnest101_training.py to train resnest_101 in these datasets
run main.py to train fusion_model

# Experimental result 



