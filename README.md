# LSF-SAC

## This paper is accepted at IEEE Transactions on Emerging Topics in Computational Intelligence (TETCI) 

Pytorch implementations of the paper [Value Functions Factorization with Latent State Information Sharing in Decentralized  Multi-Agent Policy Gradients](https://arxiv.org/abs/2201.01247) and several other multi-agent reinforcement learning algorithms, including 
[IQL](https://arxiv.org/abs/1511.08779),
[QMIX](https://arxiv.org/abs/1803.11485), [VDN](https://arxiv.org/abs/1706.05296), 
[QTRAN](https://arxiv.org/abs/1905.05408),
[QPLEX](https://arxiv.org/abs/1910.07483), [WQMIX](https://arxiv.org/abs/2006.10800), 
[DOP](https://arxiv.org/abs/2007.12322), and [COMA](https://arxiv.org/abs/1705.08926), 
which are the state of the art MARL algorithms. The paper implementation and other algorithms' implementation is based on [pymarl2](https://github.com/hijkzzz/pymarl2).

## Requirements

- python
- torch
- [SMAC](https://github.com/oxwhirl/smac)
- [pysc2](https://github.com/deepmind/pysc2)

## Acknowledgement

+ [SMAC](https://github.com/oxwhirl/smac)

## Installation
Install dependencies :

```
conda create -n pymarl python=3.8 -y
conda activate pymarl
bash install_dependencies.sh
```

Install SC2 :

```
bash install_sc2.sh
```
This will download SC2.4.10 into the 3rdparty folder and copy the maps necessary to run over.

## Run the experiments

Run all experiments
```
bash FULL_run.sh
```

or run single experiment

```
python src/main.py --config=ow_qmix --env-config=sc2 with env_args.map_name=6h_vs_8z w=0.5 epsilon_anneal_time=500000 t_max=5005000
```

All results will be stored in the ``results`` folder.

## Kill running and dead process

```
bash clean.sh
```
## Result

We independently train these algorithms for 8 times and take the mean of the 8 independent results, and we evaluate them for 20 episodes every 100 training steps. All of the results are saved in  `./result`.
Results on other maps are still in training, we will update them later.

### 1. Mean Win Rate of 8 Independent Runs with `--difficulty=7(VeryHard)`
<div align=center><img  src ="./result/GenRes_newlenged.png"/></div>


## Citation

If you find this helpful to your research, please consider citing this paper as
```
@article{zhou2022value,
  title={Value Functions Factorization with Latent State Information Sharing in Decentralized Multi-Agent Policy Gradients},
  author={Zhou, Hanhan and Lan, Tian and Aggarwal, Vaneet},
  journal={arXiv preprint arXiv:2201.01247},
  year={2022}
}
```


---
# Legacy Instruments

+ [pymarl](https://github.com/oxwhirl/pymarl)
+ [QMIX Impplementation](https://github.com/starry-sky6688/StarCraft)

The paper implementation and other algorithms' implementation is based on [starry-sky6688's qmix impplementation](https://github.com/starry-sky6688/StarCraft).

## Quick Start
```shell
$ python main.py --map=3m
```

Directly run the `main.py`, then the algorithm will start **training** on map `3m`. **Note** CommNet and G2ANet need an external training algorithm, so the name of them are like `reinforce+commnet` or `central_v+g2anet`, all the algorithms we provide are written in `./common/arguments.py`.

If you just want to use this project for demonstration, you should set `--evaluate=True --load_model=True`. 

The running of DyMA-CL is independent from others because it requires different environment settings, so we put it on another project. For more details, please read [DyMA-CL documentation](https://github.com/starry-sky6688/DyMA-CL).




## Replay

Check the website for several replay examples
[here](https://sites.google.com/view/sacmm)

If you want to see the replay from your own run, make sure the `replay_dir` is an absolute path, which can be set in `./common/arguments.py`. Then the replays of each evaluation will be saved, you can find them in your path.
