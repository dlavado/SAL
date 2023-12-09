

import ast
from datetime import datetime
from typing import List
import warnings
import torch
from torchvision import transforms
from torch import nn

# PyTorch Lightning
import pytorch_lightning as pl
import pytorch_lightning.callbacks as  pl_callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar


# Weights & Biases
import wandb
from pytorch_lightning.loggers import WandbLogger

import sys
import os


# Our code
sys.path.insert(0, '..')
sys.path.insert(1, '../..')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.models.cnn import Lit_CNN_Classifier
from core.models.resnet import LitResnet
from core.models.lit_modules.lit_wrapper import Lit_IENEONet
from core.models.vgg13 import LitVGG11



from core.data_modules.mnist import MNISTDataModule
from core.data_modules.cifar10 import CIFAR100DataModule, init_cifar10dm

from core.models.lit_modules.lit_callbaks import callback_model_checkpoint


from core.criterions.admm_loss import ADMM_Loss
from core.criterions.aug_Lag_loss import Augmented_Lagrangian_Loss
from core.criterions.constrained_loss import Constrained_Loss
from core.models.lit_modules.lit_wrapper import Lit_ADMM_Wrapper, Lit_AugLag_Wrapper, Lit_FixedPenalty_Wrapper, LitWrapperModel
from core.constraints import constraint


def init_callbacks(ckpt_dir):
    # Call back definition
    callbacks = []
    model_ckpts: List[pl_callbacks.ModelCheckpoint] = []

    ckpt_metrics = [str(met) for met in utils.init_metrics()]

    for metric in ckpt_metrics:
        model_ckpts.append(
            callback_model_checkpoint(
                dirpath=ckpt_dir,
                filename=f"val_{metric}",
                monitor=f"val_{metric}",
                mode="max",
                save_top_k=1,
                save_last=False,
                every_n_epochs=wandb.config.checkpoint_every_n_epochs,
                every_n_train_steps=wandb.config.checkpoint_every_n_steps,
                verbose=False,
            )
        )


    model_ckpts.append( # train loss checkpoint
        callback_model_checkpoint(
            dirpath=ckpt_dir, #None for default logger dir
            filename=f"val_loss",
            monitor=f"val_loss",
            mode="min",
            every_n_epochs=wandb.config.checkpoint_every_n_epochs,
            every_n_train_steps=wandb.config.checkpoint_every_n_steps,
            verbose=False,
        )
    )

    callbacks.extend(model_ckpts)

    # early_stop_callback = EarlyStopping(monitor=wandb.config.early_stop_metric, 
    #                                     min_delta=0.00, 
    #                                     patience=30, 
    #                                     verbose=False, 
    #                                     mode="min")

    # callbacks.append(early_stop_callback)

    other_callbacks = [LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)]
    callbacks.extend(other_callbacks)

    return callbacks



def init_cnn(ckpt_path=None, data_module:pl.LightningDataModule=None):
    data_module.setup()
    x = data_module.train_dataloader().dataset[0][0]
    x = x.unsqueeze(0) # adding batch dimension
    # print(f"Ghost sample shape: {x.shape}")
    # input("Press enter to continue...")
    if wandb.config.resume_from_checkpoint:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist.")
        
        print(f"Resuming from checkpoint {ckpt_path}")
        model = Lit_CNN_Classifier.load_from_checkpoint(ckpt_path,
                                                    num_classes=wandb.config.num_classes,
                                                    ghost_sample=x,
                                )
        
    else:
        model = Lit_CNN_Classifier(
            in_channels=wandb.config.in_channels,
            hidden_dim=wandb.config.hidden_dim,
            kernel_size=wandb.config.kernel_size,
            ghost_sample=x,
            num_classes=wandb.config.num_classes,
        )

    return model


def init_resnet(ckpt_path=None):
    if wandb.config.resume_from_checkpoint:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist.")
        
        print(f"Resuming from checkpoint {ckpt_path}")
        model = LitResnet.load_from_checkpoint(ckpt_path,
                                               num_classes=wandb.config.num_classes,
                                               optimizer_name=wandb.config.optimizer, 
                                               learning_rate=wandb.config.learning_rate,
                                               metric_initializer=utils.init_metrics
                                            )
        

    else:
        model = LitResnet(num_classes=wandb.config.num_classes,
                          in_channels=wandb.config.in_channels,
                          optimizer_name=wandb.config.optimizer, 
                          pretrained=wandb.config.pretrained,
                          learning_rate=wandb.config.learning_rate,
                          metric_initializer=utils.init_metrics
                        )
        
    return model

def init_vgg11(ckpt_path=None):
    if wandb.config.resume_from_checkpoint:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist.")
        
        print(f"Resuming from checkpoint {ckpt_path}")
        model = LitVGG11.load_from_checkpoint(ckpt_path,
                                               num_classes=wandb.config.num_classes,
                                               optimizer_name=wandb.config.optimizer, 
                                               learning_rate=wandb.config.learning_rate,
                                               metric_initializer=utils.init_metrics
                                            )
        

    else:
        model = LitVGG11(num_classes=wandb.config.num_classes,
                         in_channels=wandb.config.in_channels,
                         optimizer_name=wandb.config.optimizer, 
                         pretrained=wandb.config.pretrained,
                         learning_rate=wandb.config.learning_rate,
                         metric_initializer=utils.init_metrics
                        )
        
    return model


def init_ieneonet(ckpt_path=None, data_module:pl.LightningDataModule=None):
    data_module.setup()
    x = data_module.train_dataloader().dataset[0][0]
    x = x.unsqueeze(0) # adding batch dimension

    if wandb.config.resume_from_checkpoint:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist.")
        
        print(f"Resuming from checkpoint {ckpt_path}")
        model = Lit_IENEONet.load_from_checkpoint(ckpt_path,
                                                 num_classes=wandb.config.num_classes,
                                                 ghost_sample=x, 
                                                 gauss_hull_size = wandb.config.gaussian_mixture_size,   
                                                 optimizer_name=wandb.config.optimizer,
                                                 learning_rate=wandb.config.learning_rate,
                                                 metric_initializer=utils.init_metrics)
                                                  
        

    else:
        model = Lit_IENEONet(in_channels=wandb.config.in_channels,
                             hidden_dim=wandb.config.hidden_dim,
                             ghost_sample=x,
                             gauss_hull_size = wandb.config.gaussian_mixture_size,
                             kernel_size=ast.literal_eval(wandb.config.kernel_size),
                             num_classes=wandb.config.num_classes,
                             optimizer_name=wandb.config.optimizer,
                             learning_rate=wandb.config.learning_rate,
                             metric_initializer=utils.init_metrics)
        
    return model


def init_mnist(data_path, batch_size, fit_transform=None, test_transform=None):

    if fit_transform is None:
        fit_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
    
    mnist = MNISTDataModule(data_dir=data_path,
                            fit_transform=fit_transform,
                            test_transform=test_transform,
                            batch_size=batch_size,
                            num_workers=wandb.config.num_workers,)
    return mnist


def init_cifar100(data_path, batch_size):

    return CIFAR100DataModule(data_dir=data_path,
                              batch_size=batch_size)                                


def init_trainer(wandb_logger, callbacks, ckpt_path=None):
    return pl.Trainer(
            logger=wandb_logger,
            callbacks=callbacks,
            detect_anomaly=True,
            #
            max_epochs=wandb.config.max_epochs,
            accelerator=wandb.config.accelerator,
            devices=wandb.config.devices,
            #fast_dev_run = wandb.config.fast_dev_run,
            profiler=wandb.config.profiler if wandb.config.profiler else None,
            precision=wandb.config.precision,
            auto_lr_find=wandb.config.auto_lr_find,
            auto_scale_batch_size=wandb.config.auto_scale_batch_size,
            enable_model_summary=True,
            accumulate_grad_batches = ast.literal_eval(wandb.config.accumulate_grad_batches),
            resume_from_checkpoint=ckpt_path
        )


def set_admm_criterion(base_criterion, model, constraints):

    return ADMM_Loss(
        base_criterion, 
        constraints, 
        model.named_parameters(), 
        wandb.config.admm_rho,
        wandb.config.admm_rho_update_factor,
        wandb.config.admm_rho_max,
        wandb.config.admm_stepsize,
        wandb.config.admm_stepsize_update_factor
    )


def set_augLag_criterion(base_criterion, model, constraints, penalty, best_constraint_norm=None, Lag_initializer=None):

    return Augmented_Lagrangian_Loss(
        base_criterion,
        constraints,
        model.named_parameters(),
        best_constraint_norm, 
        Lag_initializer, 
        init_penalty=penalty,
        penalty_update_factor=wandb.config.admm_rho_update_factor,
        max_penalty=wandb.config.admm_rho_max
    )


def auglag_training(logger, callbacks, base_criterion, model:LitWrapperModel, constraints, data_module):

    lagrangian_multipliers = None
    best_constraint_norm = None
    convergence_iterations = wandb.config.convergence_iterations
    for k in range(convergence_iterations):
        penalty_factor = (wandb.config.admm_rho / convergence_iterations) * (k+1)

        # ------------------------
        criterion = set_augLag_criterion(base_criterion, model, constraints, penalty_factor, best_constraint_norm, lagrangian_multipliers)
        model.criterion = criterion # dynamically set up model criterion
        # ------------------------

        trainer = pl.Trainer(
            logger=logger,
            callbacks=callbacks,
            detect_anomaly=True,
            #
            max_epochs=int(wandb.config.max_epochs/convergence_iterations),
            accelerator=wandb.config.accelerator,
            devices=wandb.config.devices,
            #fast_dev_run = wandb.config.fast_dev_run,
            profiler= wandb.config.profiler if wandb.config.profiler else None,
            precision=wandb.config.precision,
            auto_lr_find=wandb.config.auto_lr_find,
            auto_scale_batch_size=wandb.config.auto_scale_batch_size,
            enable_model_summary=True,
            accumulate_grad_batches = ast.literal_eval(wandb.config.accumulate_grad_batches),
            resume_from_checkpoint=None
        )

        if wandb.config.auto_lr_find or wandb.config.auto_scale_batch_size:
            trainer.tune(model, data_module) # auto_lr_find and auto_scale_batch_size
            print(f"Learning rate in use is: {model.hparams.learning_rate}")
        
        trainer.fit(model, data_module)

        lagrangian_multipliers = criterion.get_lag_multipliers()
        best_constraint_norm = criterion.get_best_constraint_norm()

        torch.cuda.empty_cache()
        
        print(f"\nBest constraint norm : {best_constraint_norm}\n{'='*50}\n\n")

    return trainer, model


def main():

    if main_parser.opt_mode is not None:
        wandb.config.update({'convergence_mode': main_parser.opt_mode}, allow_val_change=True) # override data path

    # INIT CALLBACKS
    # --------------
  
    if not wandb.config.resume_from_checkpoint:
        ckpt_dir = os.path.join(wandb.run.dir, "checkpoints") 
    else:
        ckpt_dir = wandb.config.checkpoint_dir

    ckpt_path = os.path.join(ckpt_dir, wandb.config.resume_checkpoint_name + '.ckpt')

    callbacks = init_callbacks(ckpt_dir)


    # INIT DATA
    # ---------

    data_path = wandb.config.data_path
    data_module:pl.LightningDataModule = None

    fit_transform = utils.isomorphic_data_augmentation()
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,)),
                                         ])
    

    if wandb.config.dataset == 'mnist':
        if not os.path.exists(data_path):
            data_path = MNIST_PATH
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path {data_path} does not exist.")
        data_module = init_mnist(data_path, wandb.config.batch_size, fit_transform, test_transform)
    elif wandb.config.dataset == 'cifar10':
        if not os.path.exists(data_path):
            data_path = CIFAR10_PATH
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path {data_path} does not exist.")
        data_module = init_cifar10dm(data_path, wandb.config.batch_size)
    elif wandb.config.dataset == 'cifar100': 
        if not os.path.exists(data_path):
            data_path = CIFAR100_PATH
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path {data_path} does not exist.")
        data_module = init_cifar100(data_path, wandb.config.batch_size)   
    else:
        ValueError(f"Dataset {wandb.config.dataset} not supported.")


    # INIT MODEL
    # ----------

    constraints = None

    if wandb.config.model == 'cnn':
        model = init_cnn(ckpt_path, data_module)
        constraints = {
        'l2' : constraint.Lp_Constraint(weight=wandb.config.magnitude_weight, p=2),
        'ortho' : constraint.Orthogonality_Constraint(weight=wandb.config.ortho_weight),
        }
    elif wandb.config.model == 'resnet':
        model = init_resnet(ckpt_path)
        constraints = {
        'l1' : constraint.Lp_Constraint(weight=wandb.config.sparse_weight, p=1),
        'ortho' : constraint.Orthogonality_Constraint(weight=wandb.config.ortho_weight),
        }
    elif wandb.config.model == 'gnet':
        model = init_ieneonet(ckpt_path, data_module) 
        gmix_size = wandb.config.gaussian_mixture_size
        constraints = {
            'l2' : constraint.Lp_Constraint(weight=wandb.config.magnitude_weight, p=2),
            'ortho' : constraint.Orthogonality_Constraint(weight=wandb.config.ortho_weight, ieneo=ast.literal_eval(wandb.config.kernel_size)),
            'convex' : constraint.Convexity_Constraint(weight=wandb.config.convex_weight, cvx_coeff_name='lambda'),
            'nonneg_cvx_coeffs' : constraint.Nonnegativity_Constraint(weight=wandb.config.nonneg_weight, param_name='lambda'),
            'nonneg_stds' : constraint.Nonnegativity_Constraint(weight=wandb.config.nonneg_weight, param_name='sigmas'),
        }
    elif wandb.config.model == 'vgg11':
        model = init_vgg11(ckpt_path)
        constraints = {
        'l1' : constraint.Lp_Constraint(weight=wandb.config.sparse_weight, p=1),
        'ortho' : constraint.Orthogonality_Constraint(weight=wandb.config.ortho_weight),
        }
    else:
        ValueError(f"Model {wandb.config.model} not supported.")


    # INIT CRITERION
    # --------------
    base_criterion = nn.CrossEntropyLoss()

    if wandb.config.convergence_mode.lower() == 'admm':
        model = Lit_ADMM_Wrapper(model, None, wandb.config.optimizer, wandb.config.learning_rate, utils.init_metrics, num_classes = wandb.config.num_classes)
        criterion = set_admm_criterion(base_criterion, model, constraints)
        model.criterion = criterion # dynamically set up model criterion
    elif wandb.config.convergence_mode.lower() == 'penalty':
        model = Lit_FixedPenalty_Wrapper(model, None, wandb.config.optimizer, wandb.config.learning_rate, utils.init_metrics, num_classes = wandb.config.num_classes)
        criterion = Constrained_Loss(model.named_parameters(), base_criterion, constraints)
        model.criterion = criterion # dynamically set up model criterion
    elif wandb.config.convergence_mode.lower() == 'auglag':
        model = Lit_AugLag_Wrapper(model, None, wandb.config.optimizer, wandb.config.learning_rate, utils.init_metrics, num_classes = wandb.config.num_classes)
    else:
        ValueError(f"Convergence mode {wandb.config.convergence_mode} not supported.")

    # INIT TRAINER
    # ------------

    # WandbLogger
    wandb_logger = WandbLogger(project=f"{project_name}",
                               log_model=True, 
                               name=wandb.run.name, 
                               config=wandb.config)
    
    #wandb_logger.watch(model, log="all", log_freq=100)

    if wandb.config.convergence_mode.lower() == 'auglag':
        trainer, model = auglag_training(wandb_logger, callbacks, base_criterion, model, constraints, data_module)

    else:

        trainer = init_trainer(wandb_logger, callbacks)

        if wandb.config.auto_lr_find or wandb.config.auto_scale_batch_size:
            trainer.tune(model, data_module) # auto_lr_find and auto_scale_batch_size
            print(f"Learning rate in use is: {model.hparams.learning_rate}")

        trainer.fit(model, data_module)

    print(f"{'='*20} Model ckpt scores {'='*20}")

    for ckpt in trainer.callbacks:
        if isinstance(ckpt, pl_callbacks.ModelCheckpoint):
            if ckpt.monitor.lower() == wandb.config.resume_checkpoint_name.lower():
                ckpt_path = ckpt.state_dict()["best_model_path"]
            print(f"{ckpt.monitor} checkpoint : score {ckpt.best_model_score}")

    #wandb_logger.experiment.unwatch(model)

    # 6 TEST
    # ------

    if wandb.config.save_onnx:
        print("Saving ONNX model...")
        onnx_file_path = os.path.join(ckpt_dir, f"{project_name}.onnx")
        input_sample = next(iter(data_module.test_dataloader()))
        model.to_onnx(onnx_file_path, input_sample, export_params=True)
        wandb_logger.log({"onnx_model": wandb.File(onnx_file_path)})

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} does not exist. Using last checkpoint.")
        ckpt_path = 'best'


    trainer.test(model, 
                 datamodule=data_module,
                 ckpt_path=ckpt_path) # use the last checkpoint

    
    wandb_logger.experiment.finish()




if __name__ == '__main__':
    import pathlib
    import my_utils.utils as utils
    ROOT_PROJECT = pathlib.Path(__file__).resolve().parent.parent

    warnings.filterwarnings("ignore")

    
    if "didi" in str(ROOT_PROJECT):
        EXT_PATH = "/media/didi/TOSHIBA EXT/"
    else:
        EXT_PATH = "/home/d.lavado/" #cluster data dir

    MNIST_PATH = os.path.join(ROOT_PROJECT, "datasets", "")
    CIFAR10_PATH = os.path.join(ROOT_PROJECT, "datasets", "")
    CIFAR100_PATH = os.path.join(ROOT_PROJECT, "datasets", "")
    EXPERIMENTS_PATH = os.path.join(ROOT_PROJECT, "experiments")


    # ----------------
    main_parser = utils.main_arg_parser()
    main_parser.add_argument('--opt_mode', type=str, default=None, help='Optimization mode: admm, penalty, auglag')
    main_parser = main_parser.parse_args()

    model_name = main_parser.model
    dataset_name = main_parser.dataset

    project_name = f"ADMM_AUGLAG_{dataset_name.upper()}"

    run_name = f"{model_name}_{dataset_name}_{main_parser.opt_mode}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    experiment_path = os.path.join(EXPERIMENTS_PATH, f"{model_name}_{dataset_name}")

    os.environ["WANDB_DIR"] = os.path.abspath(experiment_path)

    print(f"\n\n{'='*100}")
    print("Entering main method...") 

    if main_parser.wandb_sweep: 
        #sweep mode
        print("wandb sweep.")
        wandb.init(project = project_name, 
                dir = experiment_path,
                name = run_name,
        )
    else:
        # default mode
        run_config = os.path.join(experiment_path, 'opt_config.yml')
        print(f"Loading config from {run_config}")

        print("wandb init.")

        wandb.init(project=project_name, 
                dir = experiment_path,
                name = run_name,
                config=run_config,
                # mode='disabled'
        )

        #pprint(wandb.config)

    main()