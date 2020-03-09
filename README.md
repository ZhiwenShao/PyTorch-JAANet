# PyTorch-JAANet
This repository is the PyTorch implementation of [JAA-Net](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhiwen_Shao_Deep_Adaptive_Attention_ECCV_2018_paper.pdf), as well as its extended version. The original Caffe implementation can be found [here](https://github.com/ZhiwenShao/JAANet)

# Getting Started
## Installation
- This code was tested with PyTorch 1.1.0 and Python 3.5
- Clone this repo:
```
git clone https://github.com/ZhiwenShao/PyTorch-JAANet
cd PyTorch-JAANet
```

## Datasets
[BP4D](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) and [DISFA](http://www.engr.du.edu/mmahoor/DISFA.htm). Put these datasets into the folder "dataset" following the paths shown in the list files of the folder "data/list"

## Preprocessing
- Conduct similarity transformation for face images:
  - Put the landmark annotation files into the folder "dataset"
  - An example of processed image can be found in the folder "data/imgs/EmotioNet/optimization_set/N_0000000001/" 
  ```
  cd dataset
  python face_transform.py
  ```
- Compute the weight of the loss of each AU in the BP4D training set:
  - The AU annoatation files should be in the folder "data/list"
```
cd dataset
python write_AU_weight.py
```

## Preprocessing
- Prepare the training data
  - Run "prep/face_transform.cpp" to conduct similarity transformation for face images
  - Run "prep/combine2parts.m" to combine two partitions as a training set, respectively
  - Run "prep/write_AU_weight.m" to compute the weight of each AU for the training set
  - Run "tools/convert_imageset" of Caffe to convert the images to leveldb or lmdb
  - Run "tools/convert_data" to convert the AU labels and weights to leveldb or lmdb: the weights are shared by all the training samples (only one line needed)
  - Our method is evaluated by 3-fold cross validation. For example, ¡°BP4D_combine_1_2¡± denotes the combination of partition 1 and partition 2
- Modify the train_val prototxt files:
  - Modify the paths of data
  - A recommended training strategy is that selecting a small set of training data for validation to choose a proper maximum iterations and then using all the training data to retrain the model

## Training
- AU detection
```
cd model
sh train_net.sh
```
- AU intensity estimation
```
sh train_net_intensity.sh
```
- Trained models on BP4D with 3-fold cross-validation for AU detection and on FERA 2015 for AU intensity estimation can be downloaded [here](https://sjtueducn-my.sharepoint.com/:f:/g/personal/shaozhiwen_sjtu_edu_cn/EsN4dd-08I9FtHnHw4bymsEB87xW7NETeW1BlIA6OS2pFw?e=Fu2HAf)

## Testing
- AU detection
```
python test.py
```
- AU intensity estimation
```
python test_intensity.py
matlab -nodisplay
>> evaluate_intensity
```
- Visualize attention maps
```
python visualize_attention_map.py
```

## Citation
If you use this code for your research, please cite our paper
```
@article{shao2019facial,
  title={Facial action unit detection using attention and relation learning},
  author={Shao, Zhiwen and Liu, Zhilei and Cai, Jianfei and Wu, Yunsheng and Ma, Lizhuang},
  journal={IEEE Transactions on Affective Computing},
  year={2019},
  publisher={IEEE}
}
```
