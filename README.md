# ECBM6040-Project
This project is the student final project for Columbia University ECBM E6040 Neural Networks and Deep Learning Research. 
This project use PyTorch to implement the LaneNet given in the the paper "Towards End-to-End Lane Detection: an Instance
Segmentation Approach". LaneNet is trained end-to-end for lane detection, by treating lane detection as an instance 
segmentation problem. 

**Image from the original paper which shows the LaneNet architecture:**
![LaneNet architecture](img/laneNet_arch.png)
___
#### Requirement
* Python 3.7
* [PyTorch 1.4.0](https://pytorch.org)
* [torchvision](https://pytorch.org/docs/stable/torchvision/index.html#torchvision)
* [sklearn 0.22.1](https://scikit-learn.org/stable/)
* [NumPy 1.18.2](https://numpy.org)

___
#### Download and prepare the dataset
**Download:**

You should download the Lane Detection Challenge dataset from [TuSimple](https://github.com/TuSimple/tusimple-benchmark/issues/3)
dataset

1. Download ``train_set.zip`` and unzip to folder ``ECBM6040-Project/TUSIMPLE``
2. Download ``test_set.zip`` and unzip to folder ``ECBM6040-Project/TUSIMPLE`` 
3. Download ``test_label.json`` and put it into the folder ``ECBM6040-Project/TUSIMPLE/test_set`` which is unzipped form ``test_set.zip``

**Prepare:**

After you download the dataset from TuSimple dataset, some preprocess to the dataset should be done to prepare the dataset 
for training and testing.

1. Process the ``train_set`` split into ground truth image, binary ground truth and instance ground truth, you should run

```
python utils/generate_tusimple_dataset.py --src_dir (your train_set folder place)
for me this step is: python utils/generate_tusimple_dataset.py --src_dir /Users/smiffy/Documents/GitHub/ECBM6040-Project/TUSIMPLE/train_set
```
2. You should see some folder like that in your ``train_set``
```
train_set
|---clips
|---label_data_0313.json
|---label_data_0531.json
|---label_data_0601.json
|---readme.md
|---training
    |---gt_binary_image
    |---gt_image
    |---gt_instance_image
    |---label_data_0313.json
    |---label_data_0531.json
    |---label_data_0601.json
    |---train.txt
```
3. Split ``train.txt`` into ``train.txt``, ``val.txt`` and ``test.txt`` put them into ``ECBM6040-Project/TUSIMPLE/txt_for_local``and re-organize folder location like that:
```
ECBM6040-Project
|---TUSIMPLE
.   |---Lanenet_output
.   |   |--lanenet_epoch_39_batch_8.model
.   |
.   |--training
.   |   |--lgt_binary_image
.   |   |--gt_image
.   |   |--gt_instance_image
.   |
.   |--txt_for_local
.   |   |--test.txt
.   |   |--train.txt
.   |   |--val.txt
.   |
.   |--test_set
.   |   |--clips
.   |   |--test_tasks_0627.json
.   |   |--test_label.json
.   |   |--readme.md
.
```
***For the data prepare you can reference [LaneNet TensorFlow project](https://github.com/MaybeShewill-CV/lanenet-lane-detection) but there is some different.***
___
#### Training the E-Net base LaneNet
1. Dataset for training: You can use ``ECBM6040-Project/Notebook-experiment/Dataset Show.ipynb`` to see the dataset for training
2. Use the ``ECBM6040-Project/Train.ipynb`` to train the LaneNet, the model will save in ``ECBM6040-Project/TUSIMPLE/Lanenet_output``
___
#### Do evaluation on the test dataset
The evaluation base on TuSimple challenge evaluation method you can get more information from [TuSimple exampe](https://github.com/TuSimple/tusimple-benchmark/blob/master/example/lane_demo.ipynb)
___
#### Generate some GIF to show the result
___
#### Reference
[1] Neven, D., De Brabandere, B., Georgoulis, S., Proesmans, M. and Van Gool, L., 2018, June. Towards end-to-end lane 
detection: an instance segmentation approach. In 2018 IEEE intelligent vehicles symposium (IV) (pp. 286-291). IEEE. 
https://arxiv.org/abs/1802.05591

[2] LaneNet TensorFlow project https://github.com/MaybeShewill-CV/lanenet-lane-detection

[3] TuSimple Dataset https://github.com/TuSimple/tusimple-benchmark

[4] E-Net Project https://github.com/davidtvs/PyTorch-ENet
