import os, time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def process(rank, world_size, fun, backend="nccl"):
    os.environ['MASTER_ADDR']="localhost"
    os.environ['MASTER_PORT']='12345'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    fun(rank, world_size)
    dist.destroy_process_group()

# simple example
def run_0(rank, world_size):
    time.sleep(2)
    print(rank, world_size)

# point 2 point communication
def run_1(rank, world_size):
    torch.cuda.set_device(rank)
    tensor = torch.zeros(1).to(rank)
    if rank == 0:
        tensor += 1
        #"向设备1发送tensor"
        dist.send(tensor=tensor, dst=1)
    if rank == 1:
        # 阻塞，等待设备0发来的tensor
        dist.recv(tensor=tensor, src=0)
    print(f"data {tensor} from rank: {rank} ")

# collective communication
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
        p = mp.Process(target=process, args=(rank, size, run_2))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    print("finished...")