import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from model.transformation import warp
from model.utils import get_network, get_transformation, get_loss_fn
from utils.misc import worker_init_fn
from lightning import LightningModule
from lightning.pytorch.loggers.logger import merge_dicts
from utils.metric import measure_metrics
from utils.visualise import visualise_results
# import sys
# sys.path.append('../..')
from data.utils_d import _apply_randconv
import matplotlib.pyplot as plt


class LightningDLReg(LightningModule):
    def __init__(self, hparams):
        super(LightningDLReg, self).__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()

        # self.train_dataset, self.val_dataset = get_datasets(self.hparams)

        self.network = get_network(self.hparams)
        self.transformation = get_transformation(self.hparams)
        self.loss_fn = get_loss_fn(self.hparams)

        # initialise dummy best metrics results for initial logging
        self.hparam_metrics = {f'hparam_metrics/{m}': 0.0
                               for m in self.hparams.hparam_metrics}
        self.validation_step_output = []

    def on_fit_start(self):
        # log dummy initial hparams w/ best metrics (for tensorboard HPARAMS)
        # self.logger.log_hyperparams(self.hparams, metrics=self.hparam_metrics)
        self.logger.log_hyperparams(self.hparams)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.data.batch_size,
                          shuffle=self.hparams.data.shuffle,
                          num_workers=self.hparams.data.num_workers,
                          pin_memory=self.on_gpu,
                          # worker_init_fn=worker_init_fn
                          )

    def val_dataloader(self):

        print('On gpu: ', self.on_gpu)
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.data.batch_size,
                          shuffle=False,
                          num_workers=self.hparams.data.num_workers,
                          pin_memory=self.on_gpu,
                          # worker_init_fn=worker_init_fn
                          )


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.training.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.hparams.training.lr_decay_step,
                                                    gamma=0.1,
                                                    last_epoch=-1)
        return [optimizer], [scheduler]

    def forward(self, tar, src):
        net_out = self.network(tar, src)
        out = self.transformation(net_out)
        return out

    def _step(self, batch):
        """ Forward pass inference + compute loss """

        if self.hparams.data.use_transformation:
            batch = _apply_randconv(batch, dim=self.hparams.data.ndim)

        tar = batch['fixed']
        src = batch['moving']
        out = self.forward(tar, src)

        if self.hparams.transformation.config.svf:
            # output the flow field and disp field
            flow, disp = out
            warped_src = warp(src, disp)
            warped_seg = warp(batch['moving_seg'], disp, interp_mode='nearest')
            # losses = self.loss_fn(tar, warped_src, flow)
            batch['warped'] = warped_src
            batch['warped_seg'] = warped_seg
            # losses = self.loss_fn(tar=batch['fixed_original'], warped_src=warped_src, u=flow)
            losses = self.loss_fn(tar=batch['fixed'], warped_src=warped_src, u=flow)

        else:
            # only output disp field
            disp = out
            warped_src = warp(src, disp)
            # losses = self.loss_fn(tar, warped_src, disp)
            batch['warped'] = warped_src
            # losses = self.loss_fn(tar=batch['fixed_original'], warped_src=warped_src, u=disp)

        step_outputs = {'disp_pred': disp,
                        'warped_moving': warped_src}
        return losses, step_outputs

    # def on_train_epoch_start(self):
    #
    #     epoch = self.global_step // 110
    #     cntr = 0
    #     if torch.remainder(torch.tensor(epoch), torch.tensor(5)) and cntr < 10:
    #         for i, param in enumerate(self.network.enc.parameters()):
    #             if (9 - i + 1) == epoch // 10 and not param.requires_grad:
    #                 param.requires_grad = True
    #                 print('\nUnfreeze layer: ', 9 - i + 1)


    def training_step(self, batch, batch_idx):
        train_losses, step_outputs = self._step(batch)
        self.log_dict({f'train_loss/{k}': loss
                       for k, loss in train_losses.items()})

        # log visualisation figure to Tensorboard
        if batch_idx == 0:
            train_data = batch
            train_data.update(step_outputs)
            # if 'moving_seg' in batch.keys():
            #     train_data['warped_moving_seg'] = warp(batch['moving_seg'], train_data['disp_pred'],
            #                                          interp_mode='nearest')
            # fig = visualise_results(train_data['fixed'][batch_idx][0].clone().cpu().detach(),
            #                         train_data['moving'][batch_idx][0].clone().cpu().detach(),
            #                         train_data['warped_moving'][batch_idx][0].clone().cpu().detach(),
            #                         train_data['disp_pred'][batch_idx].clone().cpu().detach(),
            #                         title="Training intensities", cmap='gray')
            # fig_seg = visualise_results(train_data['fixed_seg'][batch_idx][0].clone().cpu().detach(),
            #                             train_data['moving_seg'][batch_idx][0].clone().cpu().detach(),
            #                             train_data['warped_moving_seg'][batch_idx][0].clone().cpu().detach(),
            #                             train_data['disp_pred'][batch_idx].clone().cpu().detach(),
            #                             title="Training segmentation", cmap='viridis')
            # self.logger.experiment.log({f'Train intensities': fig,
            #                                "global_step": self.global_step})
            # self.logger.experiment.log({f'Trainsegmentations': fig_seg,
            #                                "global_step": self.global_step})

        return train_losses['loss']

    def validation_step(self, batch, batch_idx):
        # for k, x in batch.items():
            # reshape data for inference
            # 2d: (N=1, num_slices, H, W) -> (num_slices, N=1, H, W)
            # 3d: (N=1, 1, H, W, D) -> (1, N=1, H, W, D)
            # batch[k] = x.transpose(0, 1)

        # run inference, compute losses and outputs
        val_losses, step_outputs = self._step(batch)

        # collect data for measuring metrics and validation visualisation
        val_data = batch
        val_data.update(step_outputs)
        if 'moving_seg' in batch.keys():
            val_data['warped_moving_seg'] = warp(batch['moving_seg'], val_data['disp_pred'],
                                                 interp_mode='nearest')

        # calculate validation metrics
        val_metrics = {k: float(loss)
                       for k, loss in val_losses.items()}

        # log visualisation figure to Tensorboard
        if batch_idx == 0:
            fig = visualise_results(val_data['fixed'][batch_idx][0].clone().cpu(), val_data['moving'][batch_idx][0].clone().cpu(), val_data['warped_moving'][batch_idx][0].clone().cpu(), val_data['disp_pred'][batch_idx].clone().cpu(), title="Validation intensities", cmap='gray')
            fig_seg = visualise_results(val_data['fixed_seg'][batch_idx][0].clone().cpu(), val_data['moving_seg'][batch_idx][0].clone().cpu(), val_data['warped_moving_seg'][batch_idx][0].clone().cpu(), val_data['disp_pred'][batch_idx].clone().cpu(), title="Validation segmentation", cmap='viridis')
            # val_fig = visualise_result(val_data, axis=2)

            self.logger.experiment.log({f'Val intensities': fig,
                                           "global_step": self.global_step})
            self.logger.experiment.log({f'Val segmentations': fig_seg,
                                           "global_step": self.global_step})
            # self.logger.experiment.log({f'neurite': fig1,
            #                                "global_step": self.global_step})

            # self.logger.experiment.log({f'val_fig': val_fig,
            #                                "global_step": self.global_step})


        val_metrics.update(measure_metrics(val_data, self.hparams.metric_groups))
        self.validation_step_output.append(val_metrics)

        return val_metrics

    def on_validation_epoch_end(self):
        """ Process and log accumulated validation results in one epoch """
        val_metrics_epoch = merge_dicts(self.validation_step_output)
        self.log_dict({f'val_metrics/{k}': metric
                       for k, metric in val_metrics_epoch.items()})

        # update hparams metrics
        self.hparam_metrics = {f'hparam_metrics/{k}': val_metrics_epoch[k]
                               for k in self.hparams.hparam_metrics}
        self.validation_step_output.clear()
