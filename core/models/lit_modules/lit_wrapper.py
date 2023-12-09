from typing import Any, Iterable, Iterator, Tuple
import torch
import pytorch_lightning as pl
from torch import nn
import sys

from core.models.gnet import IENEONet



class LitWrapperModel(pl.LightningModule):
    """
    Generic Pytorch Lightning wrapper for Pytorch models that defines the logic for training, validation,testing and prediciton. 
    It also defines the logic for logging metrics and losses.    
    
    Parameters
    ----------

    `model` - torch.nn.Module:
        The model to be wrapped.
    
    `criterion` - torch.nn.Module:
        The loss function to be used

    `optimizer` - str:
        The Pytorch optimizer to be used for training.
        Note: str must be \in {'Adam', 'SGD', 'RMSprop'}

    `metric_initilizer` - function:
        A function that returns a TorchMetric object. The metric object must have a reset() and update() method.
        The reset() method is called at the end of each epoch and the update() method is called at the end of each step.
    """

    def __init__(self, model:torch.nn.Module, criterion:torch.nn.Module, optimizer_name:str, learning_rate=1e-2, metric_initializer=None, **kwargs):
        super().__init__()
        self.model = model
        self.criterion = criterion


        if metric_initializer is not None:
            self.train_metrics = metric_initializer(kwargs['num_classes'])
            self.val_metrics = metric_initializer(kwargs['num_classes'])
            self.test_metrics = metric_initializer(kwargs['num_classes'])
        else:
            self.train_metrics = None
            self.val_metrics = None
            self.test_metrics = None
    
        self.save_hyperparameters('optimizer_name', 'learning_rate')

    def forward(self, x):
        return self.model(x)
    
    def prediction(self, model_output:torch.Tensor) -> torch.Tensor:
        return model_output

    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        preds = self.prediction(out)
        
        if stage:
            on_step = stage == "train" 
            self.log(f"{stage}_loss", loss, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)

            if metric:
                met = metric(preds, y)
                self.log(f"{stage}_{metric.__name__}", met, on_epoch=True, on_step=False, prog_bar=True, logger=True)

        return loss, preds, y  

    def training_step(self, batch, batch_idx):
        loss, preds, y = self.evaluate(batch, "train")
        if self.train_metrics is not None:
            self.train_metrics(torch.flatten(preds), torch.flatten(y)).update()
        return {"loss": loss}                   
    
    def training_epoch_end(self, outputs) -> None:
        if self.train_metrics is not None:
            self._epoch_end_metric_logging(self.train_metrics, 'train', print_metrics=False)

    
    def validation_step(self, batch, batch_idx):
        loss, preds, y = self.evaluate(batch, "val")
        if self.val_metrics is not None:
            self.val_metrics(preds, y).update()
        return {"val_loss": loss}
    
    def validation_epoch_end(self, outputs) -> None: 
        if self.val_metrics is not None: # On epoch metric logging
           self._epoch_end_metric_logging(self.val_metrics, 'val', print_metrics=True)
    
    def test_step(self, batch, batch_idx):
        loss, preds, y = self.evaluate(batch, "test")
        if self.test_metrics is not None:
            self.test_metrics(preds, y).update()
        return {"test_loss": loss}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, _ = batch
        pred = self(x)
        pred = self.prediction(pred)

        return pred
    
    def test_epoch_end(self, outputs) -> None:
        if self.test_metrics is not None: # On epoch metric logging
            self._epoch_end_metric_logging(self.test_metrics, 'test')

    def get_model(self):
        return self.model
    
    def set_criteria(self, criterion):
        self.criterion = criterion
    
    def _epoch_end_metric_logging(self, metrics, prefix, print_metrics=False):
        metric_res = metrics.compute()
        if print_metrics:
            print(f'{"="*10} {prefix} metrics {"="*10}')
        for metric_name, metric_val in metric_res.items():
            if print_metrics:
                print(f'\t{prefix}_{metric_name}: {metric_val}')
            self.log(f'{prefix}_{metric_name}', metric_val, on_epoch=True, on_step=False, logger=True) 
        metrics.reset()

    def configure_optimizers(self):
        return self._resolve_optimizer(self.hparams.optimizer_name)
    
    def _check_model_gradients(self):
        print(f'\n{"="*10} Model Values & Gradients {"="*10}')
        for name, param in self.model.named_parameters():
            print(f'\t{name} -- value: {param.data.item():.5f} grad: {param.grad}')

    def _resolve_optimizer(self, optimizer_name:str):
        optimizer_name = optimizer_name.lower()
        if  optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.hparams.learning_rate)
        elif optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(self.model.parameters(), lr=self.hparams.learning_rate)
        elif optimizer_name == 'lbfgs':
            return torch.optim.LBFGS(self.model.parameters(), lr=self.hparams.learning_rate, max_iter=20)
        
        raise NotImplementedError(f'Optimizer {self.hparams.optimizer_name} not implemented')
    


class Lit_IENEONet(LitWrapperModel):
    
    def __init__(self, 
                 in_channels=1, 
                 hidden_dim=128, 
                 ghost_sample:torch.Tensor = None,
                 gauss_hull_size=5,
                 kernel_size=(3,3),
                 num_classes=10,
                 optimizer_name = 'adam', 
                 learning_rate=0.01, 
                 metric_initializer=None):

        model = IENEONet(in_channels, hidden_dim, ghost_sample, gauss_hull_size, kernel_size, num_classes)
        criterion = nn.CrossEntropyLoss() 
        super().__init__(model, criterion, optimizer_name, learning_rate, metric_initializer, num_classes=num_classes)

    def prediction(self, model_output: torch.Tensor) -> torch.Tensor:
        return torch.argmax(model_output, dim=1)
    
    def print_cvx_combination(self) -> str:
        return self.model.print_cvx_combination()
    
    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y) + self.model.nonneg_loss()
        preds = self.prediction(out)
        
        if stage:
            on_step = stage == "train" 
            self.log(f"{stage}_loss", loss, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)

            if metric:
                met = metric(preds, y)
                self.log(f"{stage}_{metric.__name__}", met, on_epoch=True, on_step=False, prog_bar=True, logger=True)

        return loss, preds, y
    
    def maintain_convexity(self):
        self.model.maintain_convexity()

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        self.maintain_convexity()
        print(self.model.print_cvx_combination())
        return super().on_train_batch_end(outputs, batch, batch_idx)


class Lit_ADMM_Wrapper(LitWrapperModel):


    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer_name: str, learning_rate=0.01, metric_initializer=None, **kwargs):
        self.lag_mult_check = False
        super().__init__(model, criterion, optimizer_name, learning_rate, metric_initializer, **kwargs)


    def prediction(self, model_output: torch.Tensor) -> torch.Tensor:
        return self.model.prediction(model_output)
    
    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        preds = self.prediction(out)

        data_fidelity = self.criterion.objective_function(out, y)
        constraint_violation = self.criterion.get_constraint_violation(self.named_parameters())
        
        if stage:
            on_step = stage == "train"
            self.log(f"{stage}_loss", loss, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)
            self.log(f"{stage}_data_fidelity", data_fidelity, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)
            self.log(f"{stage}_constraint_violation", constraint_violation, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)

            if metric:
                met = metric(preds, y)
                self.log(f"{stage}_{metric.__name__}", met, on_epoch=True, on_step=False, prog_bar=True, logger=True)


        return loss, preds, y 
    


    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        """
        After each training step, we update the contraint variables (i.e., \psi) and the lagrangian multipliers (i.e., \lambda)
        """


        constraint_violation = self.criterion.get_constraint_violation()

        if not self.lag_mult_check: # don't make the tf file huge
            if isinstance(self.model, Lit_IENEONet):
                print(f"{'='*5}> {self.model.print_cvx_combination()}")
            
            print(f"{'='*5}> Constraint Violation: {constraint_violation:.3f}")
            print(f"{'='*5}> best_constraint_norm: {self.criterion.best_constraint_norm:.3f}")
            print(f"{'='*5}> Penalty: {self.criterion.penalty_factor:.3f}")
            print(f"{'='*5}> Objective_function loss: {self.criterion.objective_function(self(batch[0]), batch[1]):.3f}")
            print(f"{'='*5}> ADMM Lagrangian loss: {self.criterion.ADMM_regularizer(self.named_parameters()):.3f}")
            print(f"{'='*5}> ADMM Step loss: {self.criterion.Stochastic_ADMM_regularizer(self.named_parameters()):.3f}")

            self.lag_mult_check = True

        with torch.no_grad():
            self.criterion.update_theta_k()
            self.criterion.update_psi()
           
            if float(self.criterion.best_constraint_norm) < constraint_violation:
                # if the constraint violation is increasing, we increase the penalty
                self.criterion.update_penalty()
            else:
                # otherwise, we increase the stepsize to accelerate convergence
                self.criterion.update_stepsize()   
                self.criterion.update_lag_multipliers()         
                
            self.criterion.current_constraint_norm = constraint_violation
            self.criterion.update_best_constraint_norm()

            # if self.criterion.ADMM_regularizer(self.named_parameters()) + self.criterion.Stochastic_ADMM_regularizer(self.named_parameters()) < self.criterion.objective_function(self(batch[0]), batch[1]):
            #     self.criterion.update_penalty(0.5)

        if isinstance(self.model, Lit_IENEONet):
            self.model.maintain_convexity()

        return super().on_train_batch_end(outputs, batch, batch_idx)

   
    
    def on_validation_epoch_end(self) -> None:
        self.lag_mult_check = False
        return super().on_validation_epoch_end() 
    
    def configure_optimizers(self):
        return self.model.configure_optimizers()
    

    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        
        constraint_violation = self.criterion.get_constraint_violation(self.named_parameters())

        self.log(f'constraint_violation', constraint_violation, on_epoch=False, on_step=True, prog_bar=False, logger=True)

        return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)
    


class Lit_AugLag_Wrapper(LitWrapperModel):

    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer_name: str, learning_rate=0.01, metric_initializer=None, **kwargs):
        self.lag_mult_check = False
        super().__init__(model, criterion, optimizer_name, learning_rate, metric_initializer, **kwargs)

    def prediction(self, model_output: torch.Tensor) -> torch.Tensor:
        return self.model.prediction(model_output)
    
    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        preds = self.prediction(out)

        data_fidelity = self.criterion.objective_function(out, y)
        constraint_violation = self.criterion.get_constraint_violation(self.named_parameters())
        
        if stage:
            on_step = stage == "train"
            self.log(f"{stage}_loss", loss, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)
            self.log(f"{stage}_data_fidelity", data_fidelity, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)
            self.log(f"{stage}_constraint_violation", constraint_violation, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)

            if metric:
                met = metric(preds, y)
                self.log(f"{stage}_{metric.__name__}", met, on_epoch=True, on_step=False, prog_bar=True, logger=True)


        return loss, preds, y 


    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        """
        After each training step, we update the contraint variables (i.e., \psi) and the lagrangian multipliers (i.e., \lambda)
        """
        #print('Update Lagrangian Multipliers and \psi...')

        with torch.no_grad():
            if self.criterion.has_constraint_norm_decreased(): 
                # then update the lagrangian multipliers
                # This entails that the constraints are better satisfied by the current model parameters
                self.criterion.update_lag_multipliers()
                self.criterion.update_best_constraint_norm() # update the best constraint norm since it has decreased
                # the penalty is  maintained since it is sufficient to lead to a decrease in the constraint norm
            else:
                # the penalty is increased since the constraint norm has not decreased
                self.criterion.increase_penalty()
                # The lagrangian multipliers are maintained since it the constraints are not better satisfied by the current model parameters  

        if isinstance(self.model, Lit_IENEONet):
            self.model.maintain_convexity()

        if not self.lag_mult_check: # don't make the tf file huge
            if isinstance(self.model, Lit_IENEONet):
                print(f"{'='*5}> {self.model.print_cvx_combination()}")
            print(f"{'='*5}> Constraint Violation: {self.criterion.get_constraint_violation():.3f}")
            print(f"{'='*5}> best_constraint_norm: {self.criterion.best_constraint_norm:.3f}")
            print(f"{'='*5}> Penalty: {self.criterion.penalty_factor:.3f}")
            print(f"{'='*5}> Objective_function loss: {self.criterion.objective_function(self(batch[0]), batch[1]):.3f}")
            print(f"{'='*5}> Lagrangian loss: {self.criterion.Lagrangian_regularizer():.3f}")
            print(f"{'='*5}> Aug Lag loss: {self.criterion.aug_Lagrangian_regularizer():.3f}")
            self.lag_mult_check = True
        return super().on_train_batch_end(outputs, batch, batch_idx)
    
    def on_validation_epoch_end(self) -> None:
        self.lag_mult_check = False
        return super().on_validation_epoch_end() 
    
    def configure_optimizers(self):
        return self.model.configure_optimizers()
    
    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        
        constraint_violation = self.criterion.get_constraint_violation()

        self.log(f'constraint_violation', constraint_violation, on_epoch=False, on_step=True, prog_bar=False, logger=True)

        return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)
    


class Lit_FixedPenalty_Wrapper(LitWrapperModel):

    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer_name: str, learning_rate=0.01, metric_initializer=None, **kwargs):
        super().__init__(model, criterion, optimizer_name, learning_rate, metric_initializer, **kwargs)

    def prediction(self, model_output: torch.Tensor) -> torch.Tensor:
        return self.model.prediction(model_output)
    
    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        preds = self.prediction(out)

        data_fidelity = self.criterion.objective_function(out, y)
        constraint_violation = self.criterion.get_constraint_violation(self.named_parameters())
        
        if stage:
            on_step = stage == "train"
            self.log(f"{stage}_loss", loss, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)
            self.log(f"{stage}_data_fidelity", data_fidelity, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)
            self.log(f"{stage}_constraint_violation", constraint_violation, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)

            if metric:
                met = metric(preds, y)
                self.log(f"{stage}_{metric.__name__}", met, on_epoch=True, on_step=False, prog_bar=True, logger=True)


        return loss, preds, y 
    
    # def on_after_backward(self) -> None:
        # for name, param in self.model.named_parameters():
        #     if 'lambda' in name:
        #         print(f"{'='*5}> {name}:\n {param.grad}")
               


    def on_validation_epoch_end(self) -> None:
        self.lag_mult_check = False
        return super().on_validation_epoch_end() 
    

    def configure_optimizers(self):
        return self.model.configure_optimizers()
    

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        if not self.lag_mult_check: # don't make the tf file huge
            if isinstance(self.model, Lit_IENEONet):
                print(f"{'='*5}> {self.model.print_cvx_combination()}")
            print(f"{'='*5}> Constraint Violation: {self.criterion.get_constraint_violation():.3f}")
            print(f"{'='*5}> Objective_function loss: {self.criterion.objective_function(self(batch[0]), batch[1]):.3f}")
            print(f"{'='*5}> Constraint Norm loss: {self.criterion._compute_constraint_norm():.3f}")

            self.lag_mult_check = True
        
        if isinstance(self.model, Lit_IENEONet):
            self.model.maintain_convexity()

        return super().on_train_batch_end(outputs, batch, batch_idx)
    

    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        
        constraint_violation = self.criterion.get_constraint_violation()

        self.log(f'constraint_violation', constraint_violation, on_epoch=False, on_step=True, prog_bar=False, logger=True)

        return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)