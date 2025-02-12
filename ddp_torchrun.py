import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# 引入分布式包
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp

# ddp setup
def ddp_setup():
    # print(os.environ) torchrun 已经将LOCAL_RANK WORLD_SIZE 参数设置在了环境变量里面
    # 初始化进程组
    init_process_group(backend='nccl') # 采用默认初始化方式（从环境变量获取LOCAL_RANK WORLD_SIZE等参数）
    torch.cuda.set_device(int(os.environ['LOCAL_RANK'])) # 设置当前进程使用的设备编号

# 随机生成数据集
class MyDataset(Dataset):
    def __init__(self, size) -> None:
        super().__init__()
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]

class Trainer():
    def __init__(self,
                 model:nn.Module,
                 train_dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer):
        # torchrun会自动将当前进程号放到os环境变量中，用户直接获取即可
        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.model = model.to(self.gpu_id) # 将模型迁移到当前进程占用的设备上
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        # 用ddp包装模型
        self.model = DDP(model, device_ids=[self.gpu_id])

    def train(self, max_epoch:int):
        for epoch in range(max_epoch):
            self._run_epoch(epoch)
    
    def _run_epoch(self, epoch):
        batch_size = len(next(iter(self.train_dataloader))[0])
        self.train_dataloader.sampler.set_epoch(epoch) # 重新设置采样器的状态，以便重新对数据集进行划分
        step=0
        for x, y in self.train_dataloader:
            x = x.to(self.gpu_id)
            y = y.to(self.gpu_id)
            self._run_batch(x, y)
            step+=1
        print(f"GPU:{self.gpu_id} | Epoch: {epoch} | Batchsize: {batch_size} | Steps: {step}")
        

    def _run_batch(self, x, y):
        self.optimizer.zero_grad()
        y_ = self.model(x)
        loss = F.cross_entropy(y_, y)
        loss.backward()
        self.optimizer.step()

def main(max_epochs:int, batch_size:int):
    ddp_setup()
    if int(os.environ['LOCAL_RANK'])==0:# 仅在0号进程上打印信息
        print(f"GPU_NUM: {torch.cuda.device_count()}")
    train_dataset = MyDataset(4096)
    # 分布式训练需要将数据（无重复）划分到不同的设备上，DistributedSampler自动帮我们完成
    td = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size,
                                  sampler=td)
    model = nn.Linear(20, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    trainer = Trainer(model, train_dataloader, optimizer)
    trainer.train(max_epochs)
    # 销毁进程组
    destroy_process_group()

# 分布式训练
if __name__=="__main__":
    import argparse # 从命令行接收参数
    parser = argparse.ArgumentParser(description="distributed training toy case")
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    main(args.max_epochs, args.batch_size)

