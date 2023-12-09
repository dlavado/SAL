
import ast
from datetime import datetime
from typing import List, Tuple
import warnings
import torch
from torchmetrics import MetricCollection
from torchmetrics.regression import mse
from torchvision import transforms
from torch import nn


import sys
import os
from tqdm import tqdm
import wandb
import re



# Our code
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from core.models.lit_modules.lit_callbaks import callback_model_checkpoint
from core.data_modules.orbit_dataset import OrbitDataModule

from core.criterions.admm_loss import ADMM_Loss, ADMM_Loss_PINN
from core.criterions.aug_Lag_loss import Augmented_Lagrangian_Loss, Augmented_Lagrangian_Loss_PINN
from core.criterions.constrained_loss import Constrained_Loss, Constrained_Loss_PINN

from core.constraints.threebody_constraint import ThreeBodyConstraint
from core.models.MLP import MLP, Lit_PINN
from core.models.lit_modules.lit_wrapper import Lit_ADMM_PINN_Wrapper, Lit_AugLag_PINN_Wrapper, Lit_FixedPenalty_PINN_Wrapper




def replace_variables(string):
    """
    Replace variables marked with '$' in a string with their corresponding values from the local scope.

    Args:
    - string: Input string containing variables marked with '$'

    Returns:
    - Updated string with replaced variables
    """
    pattern = r'\${(\w+)}'
    matches = re.finditer(pattern, string)

    for match in matches:
        variable = match.group(1)
        value = locals().get(variable)
        if value is None:
            value = globals().get(variable)

        if value is not None:
            string = string.replace(match.group(), str(value))
        else:
            raise ValueError(f"Variable '{variable}' not found.")

    return string

def save_checkpoint(model, optimizer, loss, checkpoint_dir, ckpt_name):
    """
    Save model checkpoint.

    Args:
    - model: PyTorch model to be saved
    - optimizer: Optimizer state for continuing training
    - epoch: Current epoch number
    - checkpoint_dir: Directory to save the checkpoints
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, ckpt_name)

    checkpoint = {
        'loss' : loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model checkpoint.

    Args:
    - model: PyTorch model to be loaded
    - optimizer: Optimizer for loading state_dict
    - checkpoint_path: Path to the saved checkpoint
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']

    print(f"Checkpoint loaded from {checkpoint_path}, with val_loss: {loss}")
    return model, optimizer, loss


#####################################################################
# INIT MODELS
#####################################################################

def init_metrics():
    return  MetricCollection([
       mse.MeanSquaredError()
    ])


def init_pinn():
    feat_size = wandb.config.feat_size
    window_size = wandb.config.window_size

    hidden_dim = wandb.config.hidden_dim
        
    layers = [feat_size * window_size, hidden_dim, hidden_dim, hidden_dim//2, hidden_dim//4, hidden_dim//8, feat_size - 1]  # disregard time

    model = Lit_PINN(
        layers=layers,
        batch_norm=wandb.config.batch_norm,
        dropout=wandb.config.dropout,
        optimizer_name=wandb.config.optimizer,
        learning_rate=wandb.config.learning_rate,
        metric_initializer=None,
    )
        
    return model


def init_orbit_dataset(data_path, batch_size):    
    orbit = OrbitDataModule(
        data_dir=data_path,
        window_size=wandb.config.window_size,
        batch_size=batch_size,
        num_workers=wandb.config.num_workers,
    )
    return orbit


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
    if model_name == 'pinn':
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
#####################################################################
# TRAINING
#####################################################################


def validate(model: torch.nn.Module, dataloader, val_metrics, print_metrics, best_val_loss, stage='val'):
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        running_loss = 0.0
        for inputs, labels in dataloader:
            x, time, y = model.preprocess_batch((inputs, labels))
            x, time, y = x.to(consts.device), time.to(consts.device), y.to(consts.device)

            out = model(x, time)
            preds = model.prediction(out)
            
            running_loss += base_criterion(out, y)

            if val_metrics:
                for metric_name, metric_val in val_metrics.items():
                    met = metric_val(preds.reshape(-1), y.reshape(-1))
                    if isinstance(met, torch.Tensor):
                        met = met.mean()
    
    loss = running_loss / len(dataloader)
    
    if print_metrics:
        print(f"\tRunning Loss: {loss}")
    
    wandb.log({f"{stage}_loss": loss})
    
    if loss < best_val_loss:
        best_val_loss = loss
        save_checkpoint(model, optim, loss, checkpoint_dir, checkpoint_name)
    

    if val_metrics:
        metric_result = val_metrics.compute()
        for metric_name, metric_val in metric_result.items():
            wandb.log({f"{stage}_{metric_name}": metric_val})
            if print_metrics:
                print(f"\t{metric_name}: {metric_val}")
        val_metrics.reset()
            
    model.train()  # Set the model back to training mode

    return best_val_loss



def fixed_penalty_train(model:torch.nn.Module, optimizer):
    from core.constraints.elastic_net_reg import ElasticNetRegularization
 
    model.train()

    best_val_loss = 1e10 # not the best place to put this, but it works
    first_forward = True
    cv_k = 0.0
    cv_0 = wandb.config.cv_0 # initial constraint violation

    elastic_reg = ElasticNetRegularization(alpha=wandb.config.reg_weight, l1_ratio=0.5)

    for epoch in tqdm(range(num_epochs), desc="training..."):
        running_loss = 0.0
        running_constraint_viol = 0.0

        for i, (inputs, labels) in enumerate(train_dataloader, 0):
            x, time, y = model.preprocess_batch((inputs, labels))
            x, time, y = x.to(consts.device), time.to(consts.device), y.to(consts.device)

            out = model(x, time)
            preds = model.prediction(out)

            loss_dict = criterion(model.training, out, time, y)
            loss = loss_dict['data_fidelity'] + loss_dict['constraint_enforcement']
            loss = loss + elastic_reg(model.parameters())

            loss.backward()
            optimizer.step()
            

            running_loss += loss.item()
            running_constraint_viol += loss_dict['constraint_violation'].item()
            
            if train_metrics:
                for metric_name, metric_val in train_metrics.items():
                    met = metric_val(preds.reshape(-1), y.reshape(-1))
                    if isinstance(met, torch.Tensor):
                        met = met.mean()

        cv_k = running_constraint_viol / len(train_dataloader)

        if first_forward:
            if cv_0 is None:
                cv_0 = cv_k
            first_forward = False

        print(f"Epoch {epoch + 1}\n \
              \tLoss: {running_loss / len(train_dataloader)}\n \
              \tConstraint violation: {cv_k}\n \
              \tConstraint violation ratio: {cv_k / cv_0} \
            ")
        
        wandb.log({"train_loss": running_loss / len(train_dataloader)})
        wandb.log({"train_constraint_violation": cv_k})
        wandb.log({"train_constraint_violation_ratio": cv_k / cv_0})
        
        if train_metrics:
            metric_result = train_metrics.compute()
            for metric_name, metric_val in metric_result.items():
                wandb.log({f"train_{metric_name}": metric_val})
                print(f"\t{metric_name}: {metric_val}")
            train_metrics.reset()


        # empty cache
        torch.cuda.empty_cache()

        print(f"Validating...")
        best_val_loss = validate(model, val_dataloader, val_metrics, True, best_val_loss)    
        print("") # \n

    print("Training finished.")



def admm_train(model:torch.nn.Module, optimizer):
    from core.constraints.elastic_net_reg import ElasticNetRegularization
 
    model.train()

    first_forward = True
    best_val_loss = 1e10
    cv_k = 0.0
    cv_0 = wandb.config.cv_0 # initial constraint violation

    elastic_reg = ElasticNetRegularization(alpha=wandb.config.reg_weight, l1_ratio=0.5)

    for epoch in tqdm(range(num_epochs), desc="training..."):
        running_loss = 0.0
        running_constraint_viol = 0.0

        for i, (inputs, labels) in enumerate(train_dataloader, 0):
            x, time, y = model.preprocess_batch((inputs, labels))
            x, time, y = x.to(consts.device), time.to(consts.device), y.to(consts.device)

            optimizer.zero_grad()

            out = model(x, time)
            preds = model.prediction(out)

            loss_dict = criterion(out, time, y)
            loss = loss_dict['data_fidelity'] + loss_dict['constraint_enforcement']
            loss = loss + elastic_reg(model.parameters())

            loss.backward(retain_graph=True)
            optimizer.step()
            

            running_loss += loss.item()
            running_constraint_viol += loss_dict['constraint_violation'].item()
            
            if train_metrics:
                for metric_name, metric_val in train_metrics.items():
                    met = metric_val(preds.reshape(-1), y.reshape(-1))
                    if isinstance(met, torch.Tensor):
                        met = met.mean()


        # train batch end -> update penalty / Lagrangian multipliers
        criterion.update_theta_k()
        criterion.update_psi(model, out, time)
        
        if float(criterion.best_constraint_norm) < running_constraint_viol:
            # if the constraint violation is increasing, we increase the penalty
            criterion.update_penalty()
        else:
            # otherwise, we increase the stepsize to accelerate convergence
            criterion.update_stepsize()   
            criterion.update_lag_multipliers()         
            
        criterion.current_constraint_norm = running_constraint_viol
        criterion.update_best_constraint_norm(running_constraint_viol)

        optimizer.zero_grad()

        cv_k = running_constraint_viol / len(train_dataloader)

        if first_forward:
            if cv_0 is None:
                cv_0 = cv_k
            first_forward = False

        print(f"Epoch {epoch + 1}\n \
              \tLoss: {running_loss / len(train_dataloader)}\n \
              \tConstraint violation: {cv_k}\n \
              \tConstraint violation ratio: {cv_k / cv_0} \
            ")
        
        wandb.log({"train_loss": running_loss / len(train_dataloader)})
        wandb.log({"train_constraint_violation": cv_k})
        wandb.log({"train_constraint_violation_ratio": cv_k / cv_0})
        
        if train_metrics:
            metric_result = train_metrics.compute()
            for metric_name, metric_val in metric_result.items():
                print(f"\t{metric_name}: {metric_val}")
            train_metrics.reset()


        # empty cache
        torch.cuda.empty_cache()

        print(f"Validating...")
        best_val_loss = validate(model, val_dataloader, val_metrics, True, best_val_loss)    
        print("") # \n

    print("Training finished.")



def auglag_train(model:torch.nn.Module, optimizer):
    from core.constraints.elastic_net_reg import ElasticNetRegularization
 
    model.train()

    first_forward = True
    best_val_loss = 1e10
    cv_k = 0.0
    cv_0 = wandb.config.cv_0 # initial constraint violation

    elastic_reg = ElasticNetRegularization(alpha=wandb.config.reg_weight, l1_ratio=0.5)

    lagrangian_multipliers = None
    best_constraint_norm = None
    convergence_iterations = wandb.config.convergence_iterations
    for k in range(convergence_iterations):

        penalty_factor = (wandb.config.admm_rho / convergence_iterations) * (k+1)
        model = load_checkpoint(model, optimizer, os.path.join(checkpoint_dir, checkpoint_name))[0]
        model = model.to(consts.device)

        # ------------------------
        criterion = set_augLag_criterion(base_criterion, model, constraints, penalty_factor, best_constraint_norm, lagrangian_multipliers)
        criterion = criterion.to(consts.device)
        # ------------------------
    
        # fitting step
        print(f"=== Fitting step {k+1} ===")
        for epoch in tqdm(range(int(num_epochs/convergence_iterations)), desc="training..."):
            running_loss = 0.0
            running_constraint_viol = 0.0

            for i, (inputs, labels) in enumerate(train_dataloader, 0):
                x, time, y = model.preprocess_batch((inputs, labels))
                x, time, y = x.to(consts.device), time.to(consts.device), y.to(consts.device)

                out = model(x, time)
                preds = model.prediction(out)
                
                loss_dict = criterion(model, out, time, y)
                loss = loss_dict['data_fidelity'] + loss_dict['constraint_enforcement']
                loss = loss + elastic_reg(model.parameters())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                running_loss += loss.item()
                running_constraint_viol += loss_dict['constraint_violation']
                
                if train_metrics:
                    for metric_name, metric_val in train_metrics.items():
                        met = metric_val(preds.reshape(-1), y.reshape(-1))
                        if isinstance(met, torch.Tensor):
                            met = met.mean()

            # train batch end -> update penalty / Lagrangian multipliers
            with torch.no_grad():
                if criterion.has_constraint_norm_decreased(): 
                    print(f"Constraint norm decreased; Updating Lag Multipliers")
                    # then update the lagrangian multipliers
                    # This entails that the constraints are better satisfied by the current model parameters
                    criterion.update_lag_multipliers()
                    criterion.update_best_constraint_norm() # update the best constraint norm since it has decreased
                    # the penalty is  maintained since it is sufficient to lead to a decrease in the constraint norm
                else:
                    # the penalty is increased since the constraint norm has not decreased
                    print(f"Constraint did not decrease; enforcing higher penalty")
                    criterion.increase_penalty()
                    # The lagrangian multipliers are maintained since it the constraints are not better satisfied by the current model parameters 

            optimizer.zero_grad()
            
            cv_k = running_constraint_viol / len(train_dataloader)

            if first_forward:
                if cv_0 is None:
                    cv_0 = cv_k
                first_forward = False

            print(f"Epoch {epoch + 1}\n \
                \tLoss: {running_loss / len(train_dataloader)}\n \
                \tConstraint violation: {cv_k}\n \
                \tConstraint violation ratio: {cv_k / cv_0} \
                ")
            
            wandb.log({"train_loss": running_loss / len(train_dataloader)})
            wandb.log({"train_constraint_violation": cv_k})
            wandb.log({"train_constraint_violation_ratio": cv_k / cv_0})

            if train_metrics:
                metric_result = train_metrics.compute()
                for metric_name, metric_val in metric_result.items():
                    print(f"\t{metric_name}: {metric_val}")
                train_metrics.reset()

            print(f"Validating...")
            best_val_loss = validate(model, val_dataloader, val_metrics, True, best_val_loss)    
            print("") # \n

            # empty cache
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        lagrangian_multipliers = criterion.get_lag_multipliers()
        best_constraint_norm = criterion.get_best_constraint_norm()
    
    
    print("Training finished.")



def predict_trajectory(trajectory, model,  window_size):
    """
    Predicts the trajectory using the model.

    Parameters
    ----------

    trajectory: torch.Tensor
        The trajectory to predict, shape = (trajectory_count, feat_size)

    model: torch.nn.Module
        The model to use for prediction.

    window_size: int
        The size of the prediction window.

    """
    trajectory = trajectory[:, :-1] # disregard time
    trajectory_time = trajectory[:, -1, None] # get time, shape = (trajectory_count, 1)
    pred_trajectory = torch.zeros_like(trajectory)
    
    window = trajectory[:window_size] # fill current prediction window with the first `window_size` samples of the trajectory; shape = (window_size, feat_size - 1)
    time_window = trajectory_time[:window_size] # shape = (window_size, 1)
    pred_trajectory[:window_size] = window
    
    for i in range(window_size, trajectory.shape[0]): # iterate over the remaining samples of the trajectory
        
        x = window.unsqueeze(0).to(consts.device) # add batch dimension shape = (1, window_size, feat_size)
        time = time_window.unsqueeze(0).to(consts.device) # add batch dimension shape = (1, window_size, 1)
        
        out = model(x, time)
        pred = model.prediction(out)
        
        pred_trajectory[i] = pred
        window = pred_trajectory[i - window_size + 1: i + 1]
        time_window = trajectory_time[i - window_size + 1: i + 1]

    return pred_trajectory # shape = (trajectory_count, feat_size - 1)


def predict_trajectories(model, dataloader):
    """
    Predicts the trajectories using the model.
    """

    window_size = wandb.config.window_size

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        running_loss = 0.0
        for batch_trajectory in dataloader:

            for trajectory in batch_trajectory:
                trajectory = torch.squeeze(trajectory).to(consts.device).to(torch.float32) #rem batch dim;  shape = (trajectory_count, feat_size)
                pred_trajectory = predict_trajectory(trajectory, model, window_size)
                running_loss += base_criterion(pred_trajectory.reshape(-1, 1), trajectory[:, :-1].reshape(-1, 1))
            
            # print(f"Batch Loss: {running_loss / len(batch_trajectory)}")

    print(f"Total Loss: {running_loss / len(dataloader)}")

    wandb.log({"test_loss": running_loss / len(dataloader)})

            

if __name__ == '__main__':
    import my_utils.constants as consts
    import my_utils.utils as su
    import warnings
    import yaml

    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('medium')
    
    # is cuda available
    print(f"{'='*50} CUDA available: {torch.cuda.is_available()} {'='*50}")
    # get device specs
    print(f"{'='*3}> Device specs: {torch.cuda.get_device_properties(0)}")

    main_parser = su.main_arg_parser()
    main_parser.add_argument('--opt_mode', type=str, default=None, help='Optimization mode: admm, penalty, auglag')
    main_parser = main_parser.parse_args()

    ROOT_PROJECT = consts.get_project_root()
    model_name = main_parser.model.lower()
    dataset_name = main_parser.dataset.lower()
    project_name = f"{model_name}_{dataset_name}"
    project_name = f"ADMM_AUGLAG_{project_name}"

    run_name = f"{model_name}_{dataset_name}_{main_parser.opt_mode}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    experiment_path = consts.get_experiment_path(model_name, dataset_name)
    run_config = consts.get_experiment_config_path(model_name, dataset_name)

    os.environ["WANDB_DIR"] = os.path.abspath(experiment_path)


    # # Load configuration from a YAML file
    # with open(run_config, 'r') as config_file:
    #     config = yaml.load(config_file, Loader=yaml.FullLoader)
    #     config = dict(**config)

    if main_parser.wandb_sweep: 
        #sweep mode
        print("wandb sweep.")
        wandb.init(project = project_name, 
                dir = experiment_path,
                name = run_name,
        )
    else:
        # default mode
        print(f"Loading config from {run_config}")
        print("wandb init.")

        wandb.init(project=project_name, 
                dir = experiment_path,
                name = run_name,
                config=run_config,
                mode=main_parser.wandb_mode,
        )


    
   # override config
    if main_parser.opt_mode is not None:
        wandb.config.update({'convergence_mode': f'{main_parser.opt_mode.lower()}'}, allow_val_change=True)

    # resolve checkpoint dir
    checkpoint_dir = replace_variables(wandb.config.checkpoint_dir)
    checkpoint_name = f"ckpt_val_loss_{wandb.config.convergence_mode.lower()}"
    print(f"ckpt dir: {checkpoint_dir}")

    # get hyperparameters
    batch_size = wandb.config.batch_size
    learning_rate = wandb.config.learning_rate
    num_epochs = wandb.config.max_epochs

    # ------------------------
    # 1 INIT BASE CRITERION
    # ------------------------
    base_criterion = nn.MSELoss()
    metrics_init = init_metrics

    train_metrics = metrics_init().to(consts.device)
    val_metrics = metrics_init().to(consts.device)

    # ------------------------
    # 2 INIT MODEL
    # ------------------------
    constraints = None

    if model_name == 'pinn':
        model = init_pinn()
        constraints = {
            '3body' : ThreeBodyConstraint(weight=wandb.config.threebody_weight),
        }
    else:
        ValueError(f"Model {model_name} not supported.")

    if wandb.config.convergence_mode.lower() == 'admm':
        model = Lit_ADMM_PINN_Wrapper(model, None, wandb.config.optimizer, wandb.config.learning_rate, metrics_init)
        criterion = set_admm_criterion(base_criterion, model, constraints)
        criterion = criterion.to(consts.device)
        model.criterion = criterion  # dynamically set up model criterion
    elif wandb.config.convergence_mode.lower() == 'penalty':
        model = Lit_FixedPenalty_PINN_Wrapper(model, None, wandb.config.optimizer, wandb.config.learning_rate, metrics_init)
        if wandb.config.model == 'pinn':
            criterion = Constrained_Loss_PINN(model.named_parameters(), base_criterion, constraints)
        else:
            criterion = Constrained_Loss(model.named_parameters(), base_criterion, constraints)
        criterion = criterion.to(consts.device)
        model.criterion = criterion  # dynamically set up model criterion
    elif wandb.config.convergence_mode.lower() == 'auglag':
        model = Lit_AugLag_PINN_Wrapper(model, None, wandb.config.optimizer, wandb.config.learning_rate, metrics_init)
    else:
        raise ValueError(f"Convergence mode {wandb.config.convergence_mode} not supported.")

    model = model.to(consts.device)
    print(f"\n=== Model {model_name.upper()} initialized. ===\n")

    # ------------------------
    # 3 RESOLVE OPTIM
    # ------------------------

    optim = su.resolve_optimizer(wandb.config.optimizer, model, learning_rate=learning_rate)

    # ------------------------
    # 4 INIT DATA MODULE
    # ------------------------

    data_module = init_orbit_dataset(consts.ORBIT_DATASET_PATH, batch_size)

    print(f"\n=== Data Module {dataset_name.upper()} initialized. ===\n")

    data_module.setup('fit')
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    #------------------
    # INIT TRAINER
    #------------------

    if wandb.config.convergence_mode.lower() == 'admm':
        admm_train(model, optim)
    elif wandb.config.convergence_mode.lower() == 'penalty':
        fixed_penalty_train(model, optim)
    elif wandb.config.convergence_mode.lower() == 'auglag':
        try:
            auglag_train(model, optim)
        except Exception as e:
            print("AugLag training failed. due to: ", e)
            print("Trying to load the best checkpoint...")
    else:
        raise ValueError(f"Convergence mode {wandb.config.convergence_mode} not supported.")


    # testing
    print(f"\n=== Testing {dataset_name.upper()} ===\n")
    print(f"Loading best checkpoint from {checkpoint_dir}")
    model, _, _ = load_checkpoint(model, optim, os.path.join(checkpoint_dir, checkpoint_name))
    data_module.setup('test')
    test_dataloader = data_module.test_dataloader()
    predict_trajectories(model, test_dataloader)








