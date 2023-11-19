# Motion-Prediction
3D Motion Prediction on AMASS

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
(motion-prediction) MacBook-Air:fairmotion tim$ python fairmotion/tasks/motion_prediction/training.py --save-model-path models/ --preprocessed-path data/sampled/aa/ --epochs 100
```

### Output:
```
[2023-11-19 16:27:41] [('architecture', 'seq2seq'), ('batch_size', 64), ('device', None), ('epochs', 100), ('hidden_dim', 1024), ('lr', None), ('num_layers', 1), ('optimizer', 'sgd'), ('preprocessed_path', '../data/output-sample/aa/'), ('save_model_frequency', 5), ('save_model_path', '../models/'), ('shuffle', False)]
[2023-11-19 16:27:41] Using device: cpu
[2023-11-19 16:27:41] Preparing dataset...
[2023-11-19 16:28:11] Before training: Training loss 0.01117816984122264 | Validation loss 0.013342576199742417
[2023-11-19 16:28:11] Training model...
[2023-11-19 16:28:11] Running epoch 0 | teacher_forcing_ratio=1.0
[2023-11-19 16:33:28] Training loss 0.01009094680123338 | Validation loss 0.01287927047306977 | Iterations 12
[2023-11-19 16:33:40] Validation MAE: {6: 14.479043099299663, 12: 29.284760185296342, 18: 43.98428295044982, 24: 58.75325199083912}
[2023-11-19 16:33:41] Running epoch 1 | teacher_forcing_ratio=0.98
```

