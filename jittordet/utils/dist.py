import jittor as jt


def reduce_mean(var):
    """"Obtain the mean of tensor on different GPUs."""
    if jt.world_size == 1:
        return var
    var = var.clone()
    var = var.mpi_all_reduce('mean')
    var.sync(device_sync=True)
    return var
