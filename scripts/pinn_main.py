



# File dedicates to the main function of the PINN model on the orbit dataset
# Furthermore, the PINN model on pixed penalty, augmented lagrangian and ADMM methods

import ast
from datetime import datetime
from typing import List
import warnings
import torch
from torchmetrics import MetricCollection
from torchmetrics.regression import mse
from torchvision import transforms
from torch import nn

# PyTorch Lightning
import pytorch_lightning as pl
import pytorch_lightning.callbacks as  pl_callbacks


# Weights & Biases
import wandb
from pytorch_lightning.loggers import WandbLogger

import sys
import os


# Our code
sys.path.insert(0, '..')
sys.path.insert(1, '../..')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import my_utils.utils as utils
import my_utils.constants  as constants

from core.models.lit_modules.lit_callbaks import callback_model_checkpoint
from core.data_modules.orbit_dataset import OrbitDataModule

from core.criterions.admm_loss import ADMM_Loss, ADMM_Loss_PINN
from core.criterions.aug_Lag_loss import Augmented_Lagrangian_Loss, Augmented_Lagrangian_Loss_PINN
from core.criterions.constrained_loss import Constrained_Loss, Constrained_Loss_PINN
# from core.models.lit_modules.lit_wrapper import Lit_ADMM_Wrapper, Lit_AugLag_Wrapper, Lit_FixedPenalty_Wrapper, LitWrapperModel
from core.models.lit_modules.lit_wrapper import Lit_ADMM_PINN_Wrapper, Lit_AugLag_PINN_Wrapper, Lit_FixedPenalty_PINN_Wrapper, LitWrapperModel

from core.constraints import constraint
from core.constraints.threebody_constraint import ThreeBodyConstraint
from core.models.MLP import Lit_PINN



def init_metrics():
    return  MetricCollection([
       mse.MeanSquaredError()
    ])



def init_callbacks(ckpt_dir, ckpt_metrics):
    # Call back definition
    callbacks = []
    model_ckpts: List[pl_callbacks.ModelCheckpoint] = []


    if wandb.config.model == 'pinn':
        ckpt_metrics = []

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


    return callbacks


def init_pinn():

    feat_size = wandb.config.feat_size
    window_size = wandb.config.window_size
        
    layers = [feat_size*window_size, 128, 128, 64, 32, 16, feat_size - 1] # disregard time

    model = Lit_PINN(layers=layers,
                    optimizer_name=wandb.config.optimizer, 
                    learning_rate=wandb.config.learning_rate,
                    metric_initializer=None,
                )
        
    return model


def init_orbit_dataset(data_path, batch_size):    

    orbit = OrbitDataModule(data_dir=data_path,
                            window_size=wandb.config.window_size,
                            batch_size=batch_size,
                            num_workers=wandb.config.num_workers,
                        )
    return orbit




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
            # auto_scale_batch_size=wandb.config.auto_scale_batch_size,
            enable_model_summary=True,
            accumulate_grad_batches = ast.literal_eval(wandb.config.accumulate_grad_batches),
            resume_from_checkpoint=ckpt_path
        )


def set_admm_criterion(base_criterion, model, constraints):

    if wandb.config.model == 'pinn':
        return ADMM_Loss_PINN(
            base_criterion, 
            constraints, 
            model.named_parameters(), 
            wandb.config.admm_rho,
            wandb.config.admm_rho_update_factor,
            wandb.config.admm_rho_max,
            wandb.config.admm_stepsize,
            wandb.config.admm_stepsize_update_factor
    )

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


    if wandb.config.model == 'pinn':
        return Augmented_Lagrangian_Loss_PINN(
            base_criterion,
            constraints,
            model.named_parameters(),
            best_constraint_norm, 
            Lag_initializer, 
            init_penalty=penalty,
            penalty_update_factor=wandb.config.admm_rho_update_factor,
            max_penalty=wandb.config.admm_rho_max
        )


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


    ckpt_metrics = [str(met) for met in init_metrics()]

    callbacks = init_callbacks(ckpt_dir, ckpt_metrics)


    # INIT DATA
    # ---------

    data_path = wandb.config.data_path
    data_module:pl.LightningDataModule = None

    if wandb.config.dataset == 'orbit':
        if not os.path.exists(data_path):
            data_path = constants.ORBIT_DATASET_PATH
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path {data_path} does not exist.")
        data_module = init_orbit_dataset(data_path, wandb.config.batch_size)
    else:
        ValueError(f"Dataset {wandb.config.dataset} not supported.")


    # INIT MODEL
    # ----------

    constraints = None

    if wandb.config.model == 'pinn':
        model = init_pinn()
        constraints = {
            '3body' : ThreeBodyConstraint(weight=wandb.config.threebody_weight),
        }
    else:
        ValueError(f"Model {wandb.config.model} not supported.")

    # INIT CRITERION
    # --------------

    base_criterion = nn.MSELoss()
    metrics_init = init_metrics


    if wandb.config.convergence_mode.lower() == 'admm':
        model = Lit_ADMM_PINN_Wrapper(model, None, wandb.config.optimizer, wandb.config.learning_rate, metrics_init)
        criterion = set_admm_criterion(base_criterion, model, constraints)
        model.criterion = criterion # dynamically set up model criterion
    elif wandb.config.convergence_mode.lower() == 'penalty':
        model = Lit_FixedPenalty_PINN_Wrapper(model, None, wandb.config.optimizer, wandb.config.learning_rate, metrics_init)
        if wandb.config.model == 'pinn':
            criterion = Constrained_Loss_PINN(model.named_parameters(), base_criterion, constraints)
        else:
            criterion = Constrained_Loss(model.named_parameters(), base_criterion, constraints)
        model.criterion = criterion # dynamically set up model criterion
    elif wandb.config.convergence_mode.lower() == 'auglag':
        model = Lit_AugLag_PINN_Wrapper(model, None, wandb.config.optimizer, wandb.config.learning_rate, metrics_init)
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
    torch.set_float32_matmul_precision('medium')

    
    if "didi" in str(ROOT_PROJECT):
        EXT_PATH = "/media/didi/TOSHIBA EXT/"
    else:
        EXT_PATH = "/home/d.lavado/" #cluster data dir

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
                mode=main_parser.wandb_mode,
        )

        #pprint(wandb.config)

    main()



