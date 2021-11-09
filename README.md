# Derivative-free-attack
Code for TDSC'21 paper 'Taking Care of The Discretization Problem: A Comprehensive Study of the Discretization Problem and A Black-Box Adversarial Attack in Discrete Integer Domain'.

In this repo, we provide the code to reproduce the DFA attack on ImageNet dataset.

**D**erivative **F**ree **A**ttack is a black box attack method, which could craft adversarial examples without gradient, only need to query the model.

We implement DFA based on the framework of [RACOS](https://github.com/eyounx/RACOS). We upgraded the original algorithm and improve its efficiency and scalability with several domain-specific optimizations.

## Usage

For untargeted attack, please run

`CUDA_VISIBLE_DEVICES=XX python Run_Racos.py`

For targeted attack, please run

`CUDA_VISIBLE_DEVICES=XX python Run_Racos.py --target`

You can check the detail description of parameters by

`python Run_Racos.py --help`

After executing the attack, 
we provide `racosTools.py` to analyze the results.

Please run `python racosTools.py -file=xx` with the log file name to get the average query times, 
and attack success rate.

## Citation
If you use our method in your research, please consider citing

    @ARTICLE{bu2021taking,
    author={Bu, Lei and Zhao, Zhe and Duan, Yuchao and Song, Fu},
    journal={IEEE Transactions on Dependable and Secure Computing}, 
    title={Taking Care of The Discretization Problem: A Comprehensive Study of the Discretization Problem and A Black-Box Adversarial Attack in Discrete Integer  Domain}, 
    year={2021},
    doi={10.1109/TDSC.2021.3088661}}
    
If you have any problem about the code or our paper, please feel free to contact yuchaoduann@gmail.com, zhaozhe1@shanghaitech.edu.cn, or njubulei@gmail.com.