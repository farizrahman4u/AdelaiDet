# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
from detectron2.structures.boxes import Boxes
import logging
import os
from collections import OrderedDict
import torch
from fvcore.nn.precise_bn import get_bn_modules
from torch.nn.parallel import DistributedDataParallel

from utils.models import add_mnv2_to_registry
from utils.logs import print_cnfg_as_tags, redirect_outputs, print_experiment, cnvrg_res_print
from utils.models import print_model_stats
from callbacks.cnvrg_logs import CnvrgLogger
import wandb

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger

from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
from adet.evaluation import TextEvaluator


class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader` method.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        try:
            print_model_stats(model, input_size=(550, 550), device=cfg.MODEL.DEVICE)
        except:
            print('/could not print model stats')
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
        super(DefaultTrainer, self).__init__(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = AdetCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        default_hooks = self.build_hooks()
        self.register_hooks(default_hooks)

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("adet.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def build_hooks(self):

        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),                                             # 0
            hooks.LRScheduler(self.optimizer, self.scheduler),                  # 1
            hooks.PreciseBN(                                                    # 2
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)       # 3
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():                                              # 4
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        eval_hook = hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results)
        ret.append(eval_hook)

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=self.cfg.LOG_INTERVAL))
            ret.append(CnvrgLogger(log_interval=self.cfg.CNVRG_LOG_INTERVAL))

        return ret

    def test(self, cfg, model, evaluators=None, print_results=True):
        res = super(Trainer, self).test(cfg, model, evaluators)
        if print_results:
            if len(cfg.DATASETS.TEST) > 1:
                for ds in cfg.DATASETS.TEST:
                    cnvrg_res_print(res[ds], self.iter, ds_name=ds)
            else:
                cnvrg_res_print(res, self.iter)
        return res

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        from adet.data.builtin import THINGS_ROOT_MAPPING
        assert len(cfg.DATASETS.TRAIN) == 1
        things_root = THINGS_ROOT_MAPPING[cfg.DATASETS.TRAIN[0]]
        mapper = DatasetMapperWithBasis(cfg, things_root, True)
        return build_detection_train_loader(cfg, mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "text":
            return TextEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("adet.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")

    return cfg


def test_all_checkpoints(trainer, cfg):
    model = Trainer.build_model(cfg)
    all_checkpoints = AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).get_all_checkpoint_files()
    print('Test all checkpoints:', all_checkpoints)
    for checkpoint in sorted(all_checkpoints):
        if os.path.basename(checkpoint).startswith('model_final'):
            continue
        print('Test %s' % checkpoint)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(checkpoint, resume=False)
        res = trainer.test(cfg, model, print_results=False)
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        cnvrg_res_print(res, iter=int(os.path.basename(checkpoint)[6:-4]))

def main(args):
    cfg = setup(args)

# this is my code
    redirect_outputs(os.path.join(cfg.OUTPUT_DIR, 'log.txt'))

    if comm.is_main_process():
        print_experiment()
        print_cnfg_as_tags(cfg)
        init_wandb(cfg, args)

    add_mnv2_to_registry()

# =============

    trainer = Trainer(cfg)

    if args.eval_all:
        test_all_checkpoints(trainer, cfg)
        return



    if args.eval_only:
        model = trainer.model
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = trainer.test(cfg, model, print_results=False) # d2 defaults.py
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        cnvrg_res_print(res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED: # this is the last eval
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


def get_args():
    parser = default_argument_parser()
    parser.add_argument("--eval-all", action="store_true", help="perform evaluation to al checkpoints")
    return parser.parse_args()


def init_wandb(cfg, args):
    from cnvrg import Experiment
    from utils.logs import flatten_omegaconf
    try:
        e = Experiment()
        name = e['title']
        url = e['full_href']
        id = os.path.split(url)[-1]
    except:
        name = 'debug'
        id = 0
        url = ''
    job_type = 'test' if args.eval_only else 'train'
    config = flatten_omegaconf(cfg)
    config['cnvrg_url'] = url
    wandb.init(sync_tensorboard=True,
               name=name,
               id=id,
               job_type=job_type,
               config=config,
               dir='output',
               project='BlendMask',
               resume='auto',
               group=cfg.TASK,
               )


if __name__ == "__main__":
    args = get_args()
    print("Command Line Args:", args)
    launch(
        main,
        num_gpus_per_machine=torch.cuda.device_count(),
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    wandb.finish()


# OMP_NUM_THREADS=1 python tools/train_net.py --config-file configs/BlendMask/R_50_1x.yaml --num-gpus 1 --resume OUTPUT_DIR output/blendmask_R_50_1x SOLVER.IMS_PER_BATCH 4
# OMP_NUM_THREADS=1 python tools/train_net.py --config-file configs/BlendMask/R_50_1x.yaml --num-gpus 4 --resume OUTPUT_DIR output/blendmask_R_50_1x SOLVER.IMS_PER_BATCH 16




# OMP_NUM_THREADS=1 python tools/train_net.py --config-file output/blendmask_R_50_1x/config.yaml --eval-only --num-gpus 1 OUTPUT_DIR output/blendmask_R_50_1x MODEL.WEIGHTS output/blendmask_R_50_1x/model_final.pth
