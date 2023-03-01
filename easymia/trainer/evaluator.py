"""
评估/推理器
"""

import os
import numbers
import time
import shutil

import paddle
import numpy

from easymia.utils import utils
from easymia.utils import logger
from easymia.utils import progbar

class Evaluator(object):
    """
    TBD
    """
    def __init__(self, 
                 model, 
                 val_task_loader, 
                 metrics=None, 
                 batch_size=[1, 1],
                 num_workers=0,
                 multigpu_infer=False):
        """
        TBD
        """
        self.model = model
        self.metrics = metrics
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_task_loader = val_task_loader
        self.multigpu_infer = multigpu_infer

        if self.val_task_loader is not None:
            self.val_loader, self.val_samples = self.create_loader(self.val_task_loader, self.batch_size[1])
        else:
            self.val_loader, self.val_samples = None, None

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

    def metrics_step(self, outputs, labels):
        """
        metrics step
        logits_list: list(Tensor), multiple model outputs
        labels: Tensor, fitting target
        """
        assert len(outputs) == len(self.metrics), \
            'The length of outputs should equal to the types of metric config: {} != {}.'\
                .format(len(outputs), len(self.metrics))

        assert len(outputs) == len(labels), \
            'The length of outputs should equal to labels: {} != {}.'.format(len(outputs), len(labels))

        for i, (opt, lab) in enumerate(zip(outputs, labels)):
            self.metrics[i].step(opt, lab)

    def loss_computation(self, outputs, labels, criterions, coefs):
        """
        loss计算
        """
        assert len(outputs) == len(criterions), \
            'The length of outputs should equal to the types of loss config: {} != {}.'\
                .format(len(outputs), len(criterions))

        if isinstance(labels, (list, tuple)):
            assert len(labels) == 1 or len(outputs) == len(labels), \
                'The length of outputs should equal to the labels: {} != {}.'.format(len(outputs), len(labels))
            if len(labels) == 1: labels = labels[0]
        loss_list = []
        for i in range(len(outputs)):
            if isinstance(labels, (list, tuple)):
                loss_list.append(coefs[i] * criterions[i](outputs[i], labels[i]))
            else:
                loss_list.append(coefs[i] * criterions[i](outputs[i], labels))
        return sum(loss_list)


    def evaluate(self):
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

        if local_rank == 0: 
            evaluated_sample_ids = []

        progbar_val = progbar.Progbar(target=len(self.val_loader), verbose=1, interval=10)

        with paddle.no_grad():
            for iter, data in enumerate(self.val_loader):
                fundus_img, oct_list, labels = self.unpack_data(data)
                if self.multigpu_infer or local_rank == 0:
                    outputs = self.model(fundus_img, oct_list)
                else:
                    outputs = None
                if isinstance(outputs, dict):
                    
                    outputs, labels = self.flatten_dict_output(outputs, labels)

                if nranks > 1 and self.multigpu_infer:

                    outputs = [self.gather_tensor(output) for output in outputs]
                    if labels: # is not None
                        labels = [self.gather_tensor(label) for label in labels]

                if local_rank == 0:
                    self.metrics_step(outputs, labels) # 暂存计算metrics需要的数据
                    progbar_val.update(iter + 1)
        if local_rank == 0:
            msg = "\n[EVAL] #Images: {} ".format(len(evaluated_sample_ids))
            
            for metric_obj in self.metrics:
                metric_value = metric_obj.calc()
                if isinstance(metric_value, numbers.Number):
                    msg += "{}: {:.4f}  ".format(metric_obj.name, metric_value)
                else:
                    msg += "{}: {}  ".format(metric_obj.name, metric_value)
            logger.info(msg)

            return self.metrics[0].benchmark
        else:
            return 0

    def flatten_dict_output(self, output, label=None):
        """
        将字典类型的output与label展开为list，并确保其一一对应
        output: dict {key1: [tensor1, tensor2], key2: [tensor3, tensor4, ...]}
                        |                         |
                        v                         v
        label:  dict {key1: label1,             key2: label2}
        """
        if label is None:
            return [y for x in output.values() for y in x]

        assert isinstance(output, dict) and isinstance(label, dict), \
            "Model output and Label MUST be `dict`, got {} and {}.".format(type(output), type(label))

        assert all([k in label.keys() for k in output.keys()]), \
            "All keys in output must contained in label, got {}, {}".format(list(output.keys()), list(label.keys()))

        output_collector = []
        label_collector = []

        for key in output.keys():
            opt = output[key]
            lab = label[key]

            output_collector.extend(opt)
            label_collector.extend([lab] * len(opt))

        return output_collector, label_collector

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
            fundus_img, oct_list, labels = data.get("data"), data.get("label", None), data.get("index")
        elif isinstance(data, (list, tuple)):
            if len(data) == 2:
                fundus_img, oct_list = data
                labels = None
            elif len(data) == 3:
                fundus_img, oct_list, labels = data

        if fundus_img.dtype == paddle.uint8:
            fundus_img = (fundus_img / 255.).astype("float32")

        if type(oct_list) == list:
            for i, oct_image in enumerate(oct_list):
                if oct_image.dtype == paddle.uint8:
                    oct_image = (oct_image / 255.).astype("float32")
                    oct_list[i] = oct_image
            oct_list = paddle.to_tensor(oct_list)

        else:
            oct_list = (oct_list / 255).astype("float32")
            oct_list = paddle.to_tensor(oct_list)

        if isinstance(labels, paddle.Tensor):
            labels = [labels]

        return fundus_img, oct_list, labels