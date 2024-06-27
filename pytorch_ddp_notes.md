# 分布式训练
<img src="./assets/illustration.jpg">

分布式训练大致分为：模型并行训练和数据并行训练，这里只记录 **数据并行** 训练方法。Pytorch支持两种分布式训练:**DataParallel (DP)** 和**DistributedDataParallel (DDP)**。前者实现简单，只需要在单卡训练模型上添加少量代码即可，但是仅支持单机多卡训练方式，训练过程中只开启一个进程，无法利用多核。多张GPUs中有一个master节点，负责汇总每张卡上的梯度并求梯度均值(master节点必须等待所有节点计算完毕)，计算模型参数后分发到每张卡上，这样会导致GPU利用不均衡等问题；后者启动多个进程，通常每个进程对应一个GPU模型（当然每个进程还能对应多张卡），每个进程跑一份代码，支持多机多卡训练方式，提高了训练效率。每个进程独立计算梯度，将梯度依次传递给下一个进程，之后再把从上一个进程拿到的梯度传递给下一个进程。循环n次（进程数量）之后，所有进程就可以得到全部的梯度了（Ring All Reduce）。**每个进程只跟自己上下游两个进程进行通讯，极大地缓解了参数服务器的通讯阻塞现象！**

## node, rank 基本概念

<img src="./assets/f_3.png">

* World 是包含所有的分布式训练进程的一个组

* World_size: 组的大小（通常为process数量）

* rank：进程的标志

* local_rank: 在某一结点(node)上的进程标志

* node：节点数目

* node_rank: 节点编号

* nproc_per_node：一个节点中的进程数量，一般一个进程使用一个显卡，故也通常表述为一个节中显卡的数量；

* master_addr：master节点的ip地址，也就是 rank=0 对应的主机地址。设置该参数目的是为了让其他节点知道 0 号节点的位置，这样就可以将自己训练的参数传递过去处理；

* master_port：master节点的port号，在不同的节点上master_addr和master_port的设置是一样的，用来进行通信


## 分布式实践
**分布式一定要记住你写的每一行代码是在多个进程中都要跑一遍的，有些操作是需要同步的！！！**

**pytorch_ddp_mp.py** :利用torch.multiprocessing包提供的功能启动分布式训练

**pytorch_ddp_torchrun.py** :利用torchrun提供的功能启动分布式训练

**pytorch_ddp_gpus_communication_basic.py**,**pytorch_ddp_nccl_communication.py** :分布式训练多进程通信基础


