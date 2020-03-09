# PyTorch-JAANet
This repository is the PyTorch implementation of [JAA-Net](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhiwen_Shao_Deep_Adaptive_Attention_ECCV_2018_paper.pdf), as well as its extended version. "*v1*.py" is for the ECCV version, and "*v2*.py" is for the extened versin. The original Caffe implementation can be found [here](https://github.com/ZhiwenShao/JAANet)

# Getting Started
## Installation
- This code was tested with PyTorch 1.1.0 and Python 3.5
- Clone this repo:
```
git clone https://github.com/ZhiwenShao/PyTorch-JAANet
cd PyTorch-JAANet
```

## Datasets
[BP4D](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) and [DISFA](http://www.engr.du.edu/mmahoor/DISFA.htm)

Put these datasets into the folder "dataset" following the paths shown in the list files of the folder "data/list"

## Preprocessing
- Conduct similarity transformation for face images:
  - Put the landmark annotation files into the folder "dataset". An example file "BP4D_combine_1_2_land.txt" for two images is also provided
  ```
  cd dataset
  python face_transform.py
  ```
- Compute the inter-ocular distance of each face image
  ```
  cd dataset
  python write_biocular.py
  ```
- Compute the weight of the loss of each AU for the training set:
  - The AU annoatation files should be in the folder "data/list"
  ```
  cd dataset
  python write_AU_weight.py
  ```

## Training
- Train on BP4D with the first two folds for training and the third fold for testing:
```
python train_JAAv1.py --run_name='JAAv1' --gpu_id=0 --train_batch_size=16 --eval_batch_size=28 --train_path_prefix='data/list/BP4D_combine_1_2' --test_path_prefix='data/list/BP4D_part3' --au_num=12
```
- Train on DISFA with the first two folds for training and the third fold for testing, using the the well-trained BP4D model for initialization:
```
python train_JAAv1_disfa.py --run_name='JAAv1_DISFA' --gpu_id=0 --train_batch_size=16 --eval_batch_size=32 --train_path_prefix='data/list/DISFA_combine_1_2' --test_path_prefix='data/list/DISFA_part3' --au_num=8 --pretrain_path='JAAv1_combine_1_3' --pretrain_epoch=5 
```

## Testing
- Test the models saved in different epochs:
```
python test_JAAv1.py --run_name='JAAv1' --gpu_id=0 --start_epoch=1 --n_epochs=12 --eval_batch_size=28 --test_path_prefix='data/list/BP4D_part3' --au_num=12
```
- Visualize attention maps
```
python test_JAAv1.py --run_name='JAAv1' --gpu_id=0 --pred_AU=False --vis_attention=True --start_epoch=5 --n_epochs=5 --test_path_prefix='data/list/BP4D_part3' --au_num=12
```

## Supplement
- The PyTorch implementation for the ECCV version conducts two minor revisions to make the proposed method more general:
  - The redundant cropping of attention maps is removed
  - The assembling of local feature maps uses element-wise average instead of element-wise sum
- The differences in the extended version are detailed here

## Citation
- If you use the code of the ECCV version for your research, please cite
```
@inproceedings{shao2018deep,
  title={Deep Adaptive Attention for Joint Facial Action Unit Detection and Face Alignment},
  author={Shao, Zhiwen and Liu, Zhilei and Cai, Jianfei and Ma, Lizhuang},
  booktitle={European Conference on Computer Vision},
  year={2018},
  pages={725--740},
  organization={Springer}
}
```
- If you use the code of the extended version for your research, please cite
