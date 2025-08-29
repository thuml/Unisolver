## Unisolver for Heterogeneous 2D Navier Stokes Equation (HeterNS)

![image](./figures/HeterNS_warp_v5.png)

<center><b>Figure 1.</b> Different evaluation scenarios on the HeterNS benchmark.</center>

#### Get Started
1. Install Python 3.9 For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare data. You can obtain the experiment datasets following this link.

After downloading the datasets, put the files in your local directories. 

3. Train and evaluate model. We provide the experiment scripts under the folder ./scripts/. Run the experiments as the following examples:
```
bash scripts/train.sh # For training Unisolver
bash scripts/test_coefficient.sh # For testing Unisolver on coefficient generalization experiments
bash scripts/test_force.sh # For testing Unisolver on force generalization experiments
```

Note: you need to change the dataset path in the corresponding scripts to your local dataset directories.

### Citation

If you find this repo useful, please cite our paper.

```
@inproceedings{zhouunisolver,
  title={Unisolver: PDE-Conditional Transformers Towards Universal Neural PDE Solvers},
  author={Zhou, Hang and Ma, Yuezhou and Wu, Haixu and Wang, Haowen and Long, Mingsheng},
  booktitle={Forty-second International Conference on Machine Learning}
}
```

### Contact

If you have any questions or want to use the code, please contact zhou-h23@mails.tsinghua.edu.cn.