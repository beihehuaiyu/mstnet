import os
import numbers
import time
import shutil

import paddle
import numpy as np

from easymia.trainer import infer_plugins as ip

from easymia.utils import utils
from easymia.utils import logger
from easymia.utils import progbar

class Predict:
    def __init__(self,
                 model,
                 infer_task_loader, 
                 metrics=None, 
                 infer_plugins=[],
                 batch_size=[1, 1],
                 num_workers=0,
                 multigpu_infer=False):

        self.model = model
        self.metrics = metrics
        self.infer_plugins = ip.ComposePlugins(infer_plugins)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.infer_task_loader = infer_task_loader
        self.multigpu_infer = multigpu_infer
        if self.infer_task_loader is not None:
            self.infer_loader, self.infer_samples = self.create_loader(self.infer_task_loader)
        else:
            self.infer_loader, self.infer_samples = None, None

    def create_loader(self, task_loader, batch_size=1, training=False):
        """
        python loader -> paddle loader with multi-process
        """
        if self.multigpu_infer or training:
            batch_sampler = paddle.io.DistributedBatchSampler(
                task_loader, batch_size=batch_size, shuffle=True, drop_last=False)
        else:
            batch_sampler = None

        loader = paddle.io.DataLoader(
            task_loader,
            batch_size=batch_size if batch_sampler is None else 1,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            return_list=True,
            worker_init_fn=utils.worker_init_fn
        )
        return loader, len(task_loader)

    def infer(self):
        """
        TBD
        """
        self.model.eval()

        nranks = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()

        if nranks > 1 and self.multigpu_infer:
            # Initialize parallel environment if not done.
            if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized():
                paddle.distributed.init_parallel_env()

        progbar_val = progbar.Progbar(target=len(self.infer_loader), verbose=1, interval=10)

        with paddle.no_grad():
            cache = []
            for iter, data in enumerate(self.infer_loader):
                fundus_img, oct_list, real_index = self.unpack_data(data)

                if local_rank == 0:
                    outputs, real_index = self.infer_plugins(self.model, fundus_img, oct_list, real_index)
                else:
                    outputs = None
                
                if nranks > 1 and self.multigpu_infer:
                    outputs = [self.gather_tensor(output) for output in outputs]

                if local_rank == 0:
                    cache.append([real_index, outputs])
                    progbar_val.update(iter + 1)
            self.infer_plugins.dump(cache)


    def gather_tensor(self, tensor, stack_axis=0):
        """
        多卡Tensor聚合
        """
        tensor_list = []
        paddle.distributed.all_gather(tensor_list, tensor)
        return paddle.concat(tensor_list, axis=stack_axis)

    def unpack_data(self, data):
        """
        数据解包
        """
        if isinstance(data, dict):
            fundus_img, oct_list, real_index = data.get("data"), data.get("label", None), data.get("index")
        elif isinstance(data, (list, tuple)):
            if len(data) == 2:
                fundus_img, oct_list = data
                real_index = None
            elif len(data) == 3:
                fundus_img, oct_list, real_index = data
        fundus_img = (fundus_img / 255.).astype("float32")
        fundus_img = paddle.to_tensor(fundus_img)

        if type(oct_list) == list:
            for i, oct_image in enumerate(oct_list):
                if oct_image.dtype == paddle.uint8:
                    oct_image = (oct_image / 255.).astype("float32")
                    oct_list[i] = oct_image
            oct_list = paddle.to_tensor(oct_list)

        else:
            oct_list = (oct_list / 255).astype("float32")
            oct_list = paddle.to_tensor(oct_list)

        return fundus_img, oct_list, real_index 
         