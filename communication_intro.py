import os, time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def main(rank, world_size, fun, backend="nccl"):
    os.environ['MASTER_ADDR']="localhost"
    os.environ['MASTER_PORT']='12345'
    # 初始化
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    fun(rank, world_size)
    dist.destroy_process_group()

# simple example
def run_0(rank, world_size):
    time.sleep(2)
    print(rank, world_size)

# point 2 point communication （点到点通信）
def run_1(rank, world_size):
    torch.cuda.set_device(rank)
    # 每个设备初始化一个值为0的tensor
    if rank == 0:
        tensor = torch.ones(1).to(rank)
        #"向设备1发送tensor"
        print("sent tensor from 0 to 1")
        dist.send(tensor=tensor, dst=1)
    if rank == 1:
        # 阻塞，等待设备0发来的tensor
        tensor_hold = torch.zeros(1).to(rank)
        print(f"rank {1} waiting")
        dist.recv(tensor=tensor_hold, src=0)
        # 1 号设备收到了来自 0 号设备的值
        print(f"data {tensor_hold} from rank: {rank} ")

# collective communication （组通信）
def run_2(rank, world_size):
    group = dist.new_group([0,1])
    if rank==0:
        tensor = torch.tensor([1.0,2.0,3.0])
    if rank==1:
        tensor = torch.tensor([4.0,5.0,6.0])
    tensor = tensor.to(rank)
    print(f'data before all_reduce {tensor} from rank {rank}')
    # 将所有设备（group指定的设备上的tensor聚合）
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.AVG, group=group)
    print(f'data {tensor} from rank {rank}')


if __name__=="__main__":
    size = 2
    processes = []
    # 设置多进程启动方式 fork or spawn
    mp.set_start_method("spawn")
    for rank in range(size):
        # wrap一个进程
        p = mp.Process(target=main, args=(rank, size, run_2))
        # 启动进程
        p.start()
        processes.append(p)
    # 等待所有进程终止
    for p in processes:
        p.join()

    print("finished...")