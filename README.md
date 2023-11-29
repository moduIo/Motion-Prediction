# Motion-Prediction
PyTorch implementation of Spatio-temporal Transformers (https://github.com/eth-ait/motion-transformer) for 3D Motion Prediction on AMASS.

# Project Structure
1. `models/` contains implementations of various models,
2. `utils/` contains helper functions,
3. `main.py` is the entrypoint for the program.

# Installation

## 1. Setup Conda Environment
In the root of this project run the following commands:
```
$ conda create --name motion_prediction python=3.6
$ conda activate motion_prediction
$ pip install -e fairmotion
```

## 2. AMASS Data Processing Example
1. Follow these instructions to download the AMASS dataset: https://github.com/facebookresearch/fairmotion/tree/main/fairmotion/tasks/motion_prediction
2. Run the following command (update the `output-dir` and `input-dir` appropriately):
```
(motion_prediction) MacBook-Air:fairmotion tim$ python fairmotion/tasks/motion_prediction/preprocess.py --input-dir ../data/sample/ --output-dir ../data/output-sample/ --split-dir ./fairmotion/tasks/motion_prediction/data/ --rep aa
```

### Output:
```
[2023-11-19 16:23:31] Processing training data...
[2023-11-19 16:25:08] Processed 728 sequences
[2023-11-19 16:25:08] Processing validation data...
[2023-11-19 16:25:13] Processed 38 sequences
[2023-11-19 16:25:13] Processing test data...
[2023-11-19 16:25:16] Processed 32 sequences
```

## 3. Training Example
After preprocessing, we have data in a format similar to that given in the sampled example data `data/sampled` directory.
```
(motion-prediction) MacBook-Air:fairmotion tim$ python main.py
```

### Output:
```
Training model...
src_seqs.shape=torch.Size([64, 120, 72])
 attention_seqs.shape=torch.Size([64, 120, 1536])
 output_seqs.shape=torch.Size([64, 120, 72])
Training loss 235.84820556640625 | 
```

