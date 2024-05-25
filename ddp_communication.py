import torch
import torch.distributed as dist

def dist_scatter():
    # 将 tensor list 分发到不同的设备
    rank = dist.get_rank()
    print(f"rank: {rank}")
    # 阻塞进程直到所有的进程到达这个点
    dist.barrier()
    world_size = dist.get_world_size()
    
    tensor = torch.zeros(world_size)
    before_tensor = tensor.clone()
    if rank==0:
        t_ones = torch.ones(world_size)
        t_fives = torch.ones(world_size)*5
        scatter_list = [t_ones, t_fives]
    else:
        scatter_list = None
    dist.scatter(tensor, scatter_list, src=0)
    print(f"Scatter rank : {rank} | before scatter: {before_tensor} | after scatter: {tensor}")

def dist_gather():
    # 将 tensor list 分发到不同的设备
    rank = dist.get_rank()
    print(f"rank: {rank}")
    world_size = dist.get_world_size()
    tensor = torch.ones(1)*rank
    gather_list = [torch.zeros(1) for _ in range(world_size)] if rank == 0 else None
    # 阻塞进程直到所有的进程到达这个点
    dist.barrier()
    # gather需要在所有的rank上执行，不能加if rank==0：
    dist.gather(tensor, gather_list, dst=0)
    print(f"Gather rank : {rank} | gather: {gather_list}")


def dist_broadcast():
    dist.barrier()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    src_rank = 0
    tensor = torch.tensor(world_size) if rank == src_rank else torch.zeros(1) 
    before_tensor = tensor.clone()
    dist.broadcast(tensor=tensor, src=src_rank)
    print(f"Broacast rank : {rank} | before broacast: {before_tensor} | after broacast: {tensor}")
    dist.barrier()

def dist_reduce():
    dist.barrier()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tensor = torch.tensor(rank)
    before_tensor = tensor.clone()
    dist.reduce(tensor, op=dist.ReduceOp.SUM, dst=0)
    print(f"Reduce rank : {rank} | before reduce: {before_tensor} | after reduce: {tensor}")
    dist.barrier()

def dist_all_reduce():
    dist.barrier()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank==0:
        tensor = torch.tensor([1.,2.])
    else:
        tensor = torch.tensor([2.,3.])
    before_tensor = tensor.clone()

    dist.all_reduce(tensor)
    print(f"All_Reduce rank : {rank} | before all_reduce: {before_tensor} | after all_reduce: {tensor}")
    dist.barrier()

def dist_all_gather():
    dist.barrier()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    gather_list = [torch.zeros(1) for _ in range(world_size)]
    if rank==0:
        tensor = torch.tensor([1.])
    else:
        tensor = torch.tensor([2.])

    dist.all_gather(gather_list,tensor=tensor)
    print(f"All_gather rank : {rank} | all_gather: {gather_list}")
    dist.barrier()


def dist_reduce_scatter():
    dist.barrier()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    output = torch.ones(1, dtype=torch.int64)
    tensor_list = [torch.tensor(rank*2+1), torch.tensor(rank*2+2)]
    # 将多个设备上的tensor_list相加得到[4,6],再将[4,6]分发到不同的设备。
    dist.reduce_scatter(output,tensor_list,op=dist.ReduceOp.SUM)
    dist.barrier()
    print(f"reduce scatter rank : {rank} | tensor list: {tensor_list} tensor: {output}")
    dist.barrier()



def main():
    # 初始化进程组
    dist.init_process_group("nccl")
    # 获取当前进程号
    rank = dist.get_rank()
    # local_rank = rank%torch.cuda.device_count()
    # 设置默认设备，与torch.cuda.set_device() 区分开
    torch.set_default_device(rank)

    # dist_scatter() # 将某设备上的多个值分发到值分发到多个设备
    # dist_gather() # 将多个设备上的值汇聚到指定设备
    # dist_broadcast() # 将某设备上的值复制到多个设备
    # dist_reduce() # 将所有的设备上的值按照指定的操作（求和、均值）聚合
    # dist_all_reduce() #reduce + broacaset
    # dist_all_gather() #gather + broacaset
    dist_reduce_scatter() # reduce + scatter



if __name__=="__main__":
    main()