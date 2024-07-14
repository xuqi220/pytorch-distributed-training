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
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    # 初始化进程组
    init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

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
                 optimizer: torch.optim.Optimizer,
                 gpu_id:int):
        
        self.gpu_id = gpu_id # 当前进程id（local_rank），一般一个进程对应一个计算设备
        self.model = model.to(self.gpu_id)
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.model = DDP(model, device_ids=[self.gpu_id]) # 用ddp包装模型

    def train(self, max_epoch:int):
        for epoch in range(max_epoch):
            self._run_epoch(epoch)
    
    def _run_epoch(self, epoch):
        batch_size = len(next(iter(self.train_dataloader))[0])
        self.train_dataloader.sampler.set_epoch(epoch)
        step = 0
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

def main(rank:int, world_size:int, max_epochs:int, batch_size:int):
    if rank==0:# 在0号进程上打印信息
        print(f"GPU_Count: {torch.cuda.device_count()}")
    ddp_setup(rank, world_size)
    train_dataset = MyDataset(4096) # 4096条数据
    # 分布式训练需要将数据划分（无重复）到不同的设备上，DistributedSampler自动帮我们完成
    td = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size,
                                  sampler=td)
    model = nn.Linear(20, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    trainer = Trainer(model, train_dataloader, optimizer, rank)
    trainer.train(max_epochs)
    # 销毁进程组
    destroy_process_group()

# 分布式训练
if __name__=="__main__":
    import argparse # 从命令行接收参数
    parser = argparse.ArgumentParser(description="distributed training toy case")
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    # mp.spawn包将自动提供rank参数
    mp.spawn(main, args=(world_size, args.max_epochs, args.batch_size), nprocs=world_size)

    