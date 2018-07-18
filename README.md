# DSTC7-Keras

Participating DSTC7 in 2018

### Prerequisites

Tested under the envoriment with following packages:

```
Python 3.6
tensorflow-gpu 1.6
keras-gpu 2.x
```

### Generate preprocessed data

Make following directories under the repository

```
> cd dstc7-keras
> mkdir data
> mkdir train_data
> mkdir valid_data
```

And place following files under dstc7-keras/data:

```
1. glove.42B.300d.txt
2. ubuntu_train_subtask_1.json
3. ubuntu_dev_subtask_1.json
```

Execute following command:

```
> python3 preprocess/prepare_data.py --train_in data/ubuntu_train_subtask_1.json --validation_in data/ubuntu_dev_subtask_1.json --vocab_path data/ubuntu_subtask_1.txt  --vocab_processor data/ubuntu_subtask_1.bin
```

### Train a model

```
> python3 train.py
```
