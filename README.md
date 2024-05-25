# 分布式训练
该仓库记录基于pytorch提供的DDP以及DeepSpeed的分布式训练的学习笔记

## 环境

```
torch==2.1.0
```
本实验的数据是随机生成的4096条数据、模型只有一层dense层。在单机多卡（4 * A100-80G GPU）上进行的，当然也支持内存更小的多卡机器。本实验的代码结构简单，提供了注释，可读性强。

## MP
利用pytorch提供的torch.multiprocessing 包启动多进程执行训练脚本
```
python mp.py --max_epochs=2 --batch_size=32

[W socket.cpp:663] [c10d] The client socket has failed to connect to [localhost]:12345 (errno: 101 - Network is unreachable).
GPU_Count: 4
GPU:0 | Epoch: 0 | Batchsize: 32 | Steps: 32
GPU:2 | Epoch: 0 | Batchsize: 32 | Steps: 32
GPU:1 | Epoch: 0 | Batchsize: 32 | Steps: 32
GPU:3 | Epoch: 0 | Batchsize: 32 | Steps: 32
GPU:0 | Epoch: 1 | Batchsize: 32 | Steps: 32
GPU:2 | Epoch: 1 | Batchsize: 32 | Steps: 32
GPU:3 | Epoch: 1 | Batchsize: 32 | Steps: 32
GPU:1 | Epoch: 1 | Batchsize: 32 | Steps: 32
```
计算： 

Batchsize\*step\*GPU_NUM = 32\*32\*4 = 4096

## torchrun or launch
利用pytorch提供的torchrun 或 torch.distributed.launch 启动多进程执行训练脚本
```
torchrun --nproc-per-node=4 start_with_torchrun.py --max_epoch=2 --batch_size=32

GPU_Count: 4
GPU:2 | Epoch: 0 | Batchsize: 32 | Steps: 32
GPU:3 | Epoch: 0 | Batchsize: 32 | Steps: 32
GPU:1 | Epoch: 0 | Batchsize: 32 | Steps: 32
GPU:0 | Epoch: 0 | Batchsize: 32 | Steps: 32
GPU:3 | Epoch: 1 | Batchsize: 32 | Steps: 32
GPU:2 | Epoch: 1 | Batchsize: 32 | Steps: 32
GPU:1 | Epoch: 1 | Batchsize: 32 | Steps: 32
GPU:0 | Epoch: 1 | Batchsize: 32 | Steps: 32

*****************************************

python -m torch.distributed.launch --use-env --nproc-per-node=4 start_with_torchrun.py --max_epoch=2 --batch_size=32

GPU_Count: 4
GPU:0 | Epoch: 0 | Batchsize: 32 | Steps: 32
GPU:2 | Epoch: 0 | Batchsize: 32 | Steps: 32
GPU:1 | Epoch: 0 | Batchsize: 32 | Steps: 32
GPU:3 | Epoch: 0 | Batchsize: 32 | Steps: 32
GPU:0 | Epoch: 1 | Batchsize: 32 | Steps: 32
GPU:2 | Epoch: 1 | Batchsize: 32 | Steps: 32
GPU:1 | Epoch: 1 | Batchsize: 32 | Steps: 32
GPU:3 | Epoch: 1 | Batchsize: 32 | Steps: 32
```
torchrun 命令 等价于 python -m torch.distributed.launch --use-env

torchrun 将'LOCAL_RANK'设置环境变量中，用户需要从`os.environ('LOCAL_RANK')`中取。



## 进程通信

`ddp_communication_intro.py`
* dist.send() # 向指定设备发送tensor
* dist.recv() # 接收指定设备发送的tensor
* dist.all_reduce() # 

`ddp_communication.py`

## 自动混合精度（AMP）
