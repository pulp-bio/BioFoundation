#*----------------------------------------------------------------------------*
#* Copyright (C) 2025 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Danaé Broustail                                                   *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*
import torch
import torch.distributed as dist
import pytorch_lightning as pl
import hydra
import torch_optimizer as torch_optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import wandb
from einops import rearrange, repeat
from criterion.query_specialization_criterion import QuerySpecializationCriterion
from sklearn.manifold import TSNE
from models.modules.channel_embeddings import CHANNEL_IDX_TO_NAMES
import psutil, os
import gc

def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()

def get_world_size():
    return dist.get_world_size() if is_dist_initialized() else 1

class ChannelWiseNormalize:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, tensor):
        with torch.no_grad():
            # tensor: (B, C, T)
            mean = tensor.mean(dim=2, keepdim=True)
            std = tensor.std(dim=2, keepdim=True)
            return (tensor - mean) / (std + self.eps)

class MaskTask(pl.LightningModule):
    """
    PyTorch Lightning module for training a model with masked reconstruction.
    Args:
        hparams (DictConfig): Parameters and configurations loaded via Hydra.
    """
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = hydra.utils.instantiate(self.hparams.model)

        self.criterion = hydra.utils.instantiate(self.hparams.criterion)
        self.query_specialization_criterion = QuerySpecializationCriterion(**self.hparams.query_specialization_criterion)
        self.patch_size = self.hparams.masking.patch_size
        self.masking_ratio = self.hparams.masking.masking_ratio
        self.unmasked_loss_coeff = self.hparams.masking.unmasked_loss_coeff
        # Enable normalization if specified in parameters
        self.normalize = False
        if self.hparams.input_normalization is not None and self.hparams.input_normalization.normalize:
            self.normalize = True
            self.normalize_fct = ChannelWiseNormalize()

        self.use_lejepa = self.hparams.le_jepa.use_lejepa
        self.use_lejepa_only = self.hparams.le_jepa.use_lejepa_only
        if self.use_lejepa:
            self.num_global_views = self.hparams.le_jepa.num_global_views
            self.num_local_views = self.hparams.le_jepa.num_local_views
            self.lambd_lejepa = self.hparams.le_jepa.lambd_lejepa
            self.num_patches_local = self.hparams.le_jepa.num_patches_local
            self.num_patches_global = self.hparams.le_jepa.num_patches_global
            self.patch_width_local = self.model.patch_size * self.num_patches_local # 40 * 4 = 160 out of 1280 time steps
            self.patch_width_global = self.model.patch_size * self.num_patches_global # 40 * 16 = 640 out of 1280 time steps
            self.num_slices = self.hparams.le_jepa.num_slices  # number of slices for SIGReg
            self.lejepa_scaling_factor = self.hparams.le_jepa.lejepa_scaling_factor

    def generate_mask(self, batch_size, C, T):
        """
        Generate a boolean mask for block-wise rectangular masking.
        Args:
            batch_size (int): Batch size.
            C (int): Number of channels (height).
            T (int): Temporal length (width).
        Returns:
            torch.BoolTensor: Boolean mask of shape (batch_size, C, T),
                              with True in the masked regions.
        """
        patch_H, patch_W = self.patch_size
        masking_ratio = self.masking_ratio

        # Calculate total number of patch rectangles
        num_rectangles = (C // patch_H) * (T // patch_W)
        num_to_mask = int(num_rectangles * masking_ratio)

        row_indices = torch.arange(0, C, patch_H)
        col_indices = torch.arange(0, T, patch_W)
        rectangles = [(i, j) for i in row_indices for j in col_indices]

        # Randomly select which rectangles to mask
        selected_indices = torch.randperm(num_rectangles)[:num_to_mask]

        mask = torch.zeros(batch_size, C, T, dtype=torch.bool).to(self.device)

        # Set mask to True in the selected regions
        for idx in selected_indices:
            r, c = rectangles[idx]
            mask[:, r:r + patch_H, c:c + patch_W] = True  

        return mask

    def generate_fake_mask(self, batch_size, C, T):
        """
        Create a dummy mask tensor to simulate attention masking.
        Args:
            batch_size (int): Number of samples.
            C (int): Number of channels.
            T (int): Temporal dimension.
        
        Returns:
            torch.Tensor: Boolean mask tensor of shape (B, C, T).
        """
        return torch.zeros(batch_size, C, T, dtype=torch.bool).to(self.device)

    def sigreg_2d_projection(self, x, global_step):
        """
        Returns 2D projected samples used for logging visualization.
        """
        device = x.device
        g = torch.Generator(device=device)
        g.manual_seed(global_step)

        M = x.size(1)
        A = torch.randn((M, 2), generator=g, device=device)
        A /= (A.norm(dim=0) + 1e-12)

        with torch.no_grad():
            proj2d = (x @ A).detach().cpu()

        print("proj2d shape", proj2d.shape)
        print("Logging - A preview:", A[:2, :2])

        return proj2d  # (N, 2)

    def log_scatter_2D_SigREG(self, proj2d):
        """
        proj2d: tensor or numpy, shape (N, 2)
        return_image: if True → return HWC numpy image for logger
        
        Returns:
            np.ndarray HWC image if return_image=True
        """

        # Convert to numpy
        pts = proj2d.detach().to(torch.float32).cpu().numpy() if hasattr(proj2d, "detach") else proj2d

        fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.7)

        tag = "SIGReg 2D Projection (Epoch {})".format(self.current_epoch)
        ax.set_title(tag)
        ax.set_xlabel("slice dim 1")
        ax.set_ylabel("slice dim 2")

        # Force equal aspect ratio
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        plt.show()

        # ---------- Logging ----------
        if hasattr(self.logger.experiment, 'log'):  # wandb
            self.logger.experiment.log({
                f"LeJEPA_2D_batch_0": wandb.Image(fig),
                "epoch": self.current_epoch
            })
        elif hasattr(self.logger.experiment, 'add_figure'):  # TensorBoard
            tag = f"LeJEPA_2D_batch_0"
            self.logger.experiment.add_figure(tag, fig,
                                              global_step=self.current_epoch)

        # ---------- Save as NPZ ----------
        output_dir = os.path.join(self.logger.log_dir, "SIGReg_2D_npy")
        os.makedirs(output_dir, exist_ok=True)

        npz_filename = os.path.join(
            output_dir,
            f"SIGReg_2D_epoch_{self.current_epoch}.npz"
        )

        np.savez(
            npz_filename,
            points_2d=pts.astype(np.float32),
            epoch=self.current_epoch,
            xlabel="slice dim 1",
            ylabel="slice dim 2",
            title=f"SIGReg 2D Projection (Epoch {self.current_epoch})",
        )

        print(f"Saved SIGReg 2D projection to {npz_filename}")
        plt.close(fig)

    def SIGReg(self, x, global_step, num_slices=128):
        """
        Computes the Sliced Epps–Pulley statistic in a DDP-safe manner.
        x: (N, M) real tensor
        """
        device = x.device

        # -------- 1. Random projections (synchronized via global_step) --------
        g = torch.Generator(device=device)
        g.manual_seed(global_step)

        M = x.size(1)
        A = torch.randn((M, num_slices), generator=g, device=device)
        A /= (A.norm(dim=0) + 1e-12)

        # Project sample: (N, num_slices)
        x_proj = x @ A

        # -------- 2. Empirical characteristic function (complex) -------------
        t = torch.linspace(-5, 5, 17, device=device)
        exp_f = torch.exp(-0.5 * t**2) 

        # x_proj: (N, S) → (N, S, 17)
        x_t = x_proj.unsqueeze(2) * t

        # (N, S, 17) → (S, 17) complex
        ecf_local = (1j * x_t).exp().mean(dim=0)

        # -------- 3. All-reduce complex ECF ---------------------------
        # Convert complex → real view: (S, 17, 2)
        ecf_real = torch.view_as_real(ecf_local)

        # Sum across ranks
        if is_dist_initialized():
            dist.all_reduce(ecf_real, op=dist.ReduceOp.SUM)
            ecf_real /= get_world_size()

        # Convert back to complex: (S, 17)
        ecf = torch.view_as_complex(ecf_real)

        # -------- 4. Epps–Pulley error --------------------------------------
        err = (ecf - exp_f).abs().square() * exp_f

        # -------- 5. Compute true global N ----------------------------------
        # sum N over devices
        local_N = torch.tensor([x.size(0)], device=device, dtype=torch.long)

        if is_dist_initialized():
            dist.all_reduce(local_N, op=dist.ReduceOp.SUM)
        total_N = float(local_N.item())

        # -------- 6. Compute Epps-Pulley statistic ----------------------------------
        # ∫ err(t) dt * N
        T = torch.trapz(err, t, dim=1) * total_N

        T_statistic = T.mean()

        return T_statistic

    def log_lejepa_views(self, x, channel_names, global_views, all_views, start_times_global, start_times_local):

        # iterate over channels
        sample_idx = 9
        sample_X = x[sample_idx].detach().cpu().numpy()  # (C, T)
        sample_channel_names = channel_names[sample_idx].detach().cpu().numpy()  # (C,)
        channels_to_plot = sample_channel_names.tolist()[:5] # first 5 channels

        written_channel_names = [CHANNEL_IDX_TO_NAMES[idx] for idx in channels_to_plot]

        fig, axes = plt.subplots(len(channels_to_plot), 1, figsize=(15, 2*len(channels_to_plot)), sharex=True, dpi=150)
        time = np.arange(sample_X.shape[1]) / 256  # assuming 256 Hz
        for i, ax in enumerate(axes):
            ax.plot(time, sample_X[i])

            # shaded areas for global views
            for gv in range(self.num_global_views):
                start_t_global = start_times_global[gv][sample_idx]
                end_t_global = start_t_global + self.patch_width_global
                ax.axvspan(start_t_global / 256, end_t_global / 256, color='red', alpha=0.2)

            for lv in range(self.num_local_views):
                start_t_local = start_times_local[lv][sample_idx]
                end_t_local = start_t_local + self.patch_width_local
                ax.axvspan(start_t_local / 256, end_t_local / 256, color='green', alpha=0.2)

            ax.set_ylabel(written_channel_names[i])
            ax.grid()

        axes[-1].set_xlabel("Time (s)")
        plt.title(f"LeJEPA Views for Sample Index {sample_idx} (Red: Global Views, Green: Local Views)")
        plt.tight_layout()
        plt.show()

         # ---------- Logging ----------
        if hasattr(self.logger.experiment, 'log'):  # wandb
            self.logger.experiment.log({
                f"LeJEPA/global_local_views_batch_0_{sample_idx}": wandb.Image(fig),
                "epoch": self.current_epoch
            })
        elif hasattr(self.logger.experiment, 'add_figure'):  # TensorBoard
            tag = f"LeJEPA/global_local_views_batch_0_{sample_idx}"
            self.logger.experiment.add_figure(tag, fig,
                                              global_step=self.current_epoch)

        plt.close(fig)

    def generate_lejepa_views(self, x, num_global_views=2, num_local_views=8):
        """
        Generate global and local views for LeJEPA training.
        Args:
            X (torch.Tensor): Input batch of shape (B, C, T).   
        Returns:
            global_views (list[torch.Tensor]): List of global view tensors.
            all_views (list[torch.Tensor]): List of all view tensors (global + local).
        """
        B, C, T = x.shape

        all_views = []
        start_times_local = []
        for _ in range(num_local_views):
            start_t = torch.randint(0, T - self.patch_width_local + 1, (B,)).tolist()
            local_view = torch.stack([x[i, :, start_t[i]:start_t[i] + self.patch_width_local] for i in range(B)], dim=0)
            all_views.append(local_view)
            start_times_local.append(start_t)

        global_views = []
        start_times_global = []
        for _ in range(num_global_views):  # Example scales for global views
            start_t = torch.randint(0, T - self.patch_width_global + 1, (B,)).tolist()
            global_view = torch.stack([x[i, :, start_t[i]:start_t[i] + self.patch_width_global] for i in range(B)], dim=0)
            global_views.append(global_view)
            start_times_global.append(start_t)

        return global_views, all_views, start_times_global, start_times_local

    def LeJEPA(self, x, channel_locations, global_views, all_views, lambd, global_step, train_mode=False):
        """
            global_views and all_views are lists of
            tensors, lambd is a scalar
        """
        B, C, T = x.shape
        bs = B

        # make list of torch tensors to tensor of shape (V, B, L, D)
        V_local = len(all_views)
        V_global = len(global_views)

        all_views_tensor = torch.stack(all_views)  # (V_local, B, C, T)
        all_views_tensor = rearrange(all_views_tensor, 'V B C T -> (B V) C T', B=B)  # (B*V_local, C, T)

        global_views_tensor = torch.stack(global_views)  # (V_global, B, C, T)
        global_views_tensor = rearrange(global_views_tensor, 'V B C T -> (B V) C T', B=B)  # (B*V_global, C, T)

        channel_locations_expanded_local = channel_locations.unsqueeze(1).repeat(1, V_local, 1, 1)
        channel_locations_expanded_local = channel_locations_expanded_local.view(B * V_local, C, 3) # B*V_local, C, 3

        channel_locations_expanded_global = channel_locations.unsqueeze(1).repeat(1, V_global, 1, 1)
        channel_locations_expanded_global = channel_locations_expanded_global.view(B * V_global, C, 3) # B*V_global, C, 3

        a_emb = self.model.encode(all_views_tensor, channel_locations=channel_locations_expanded_local) # (B*V_local, L, D)
        g_emb = self.model.encode(global_views_tensor, channel_locations=channel_locations_expanded_global) # (B*V_global, L, D)

        D = a_emb.size(-1)
        K = D

        # compute mean along L dimension
        a_emb = a_emb.mean(dim=1)  # (BV, D)
        g_emb = g_emb.mean(dim=1)  # (BV, D)

        # rearrange to (V, B, D)
        a_emb = rearrange(a_emb, '(B V) D -> V B D', B=bs, V=V_local)
        g_emb = rearrange(g_emb, '(B V) D -> V B D', B=bs, V=V_global) 
        
        # compute mean of global views and similarity loss
        centers = g_emb.mean(0)

        sim = (centers - a_emb).square().mean() # scalar

        # each emb is (B, D), each sigreg is a scalar
        sigreg = torch.stack([self.SIGReg(emb, global_step, num_slices=self.num_slices) for emb in a_emb]).mean()
    
        if train_mode:
            self.log('LeJEPA_train_similarity_loss', sim, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('LeJEPA_train_SIGReg_loss', sigreg, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # final leJEPA loss
        return (1-lambd)*sim + lambd*sigreg, a_emb

    def training_step(self, batch, batch_idx):
        """
        Training step: apply mask, normalize and compute loss.
        Args:
            batch (torch.Tensor): Input batch.
            batch_idx (int): Batch index.
        Returns:
            torch.Tensor: Loss value.
        """
        X = batch["input"]
        channel_locations = batch["channel_locations"]
        channel_names = batch.get("channel_names", None)
        mask = self.generate_mask(X.shape[0], X.shape[1], X.shape[2])

        if self.normalize:
            X = self.normalize_fct(X)

        # Pass masked input through the model to get reconstruction and embeddings
        x_reconstructed, x_original, attention_scores = self.model(X, mask, channel_locations, channel_names)

        # Compute loss only on masked parts
        if not self.use_lejepa_only:
            masked_loss, unmasked_loss = self.criterion(x_reconstructed, x_original, mask)
            loss = masked_loss + self.unmasked_loss_coeff * unmasked_loss
            self.log('masked_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            if self.hparams.query_specialization_criterion is not None:
                query_specialization_loss = self.query_specialization_criterion(attention_scores)
                loss += query_specialization_loss
                self.log('query_specialization_loss', query_specialization_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        else: # use only LeJEPA loss
            loss = torch.tensor(0.0, device=self.device)

        # Compute LeJEPA loss and add to total loss
        if self.use_lejepa:
            global_views, all_views, start_times_global, start_times_local = self.generate_lejepa_views(X,
                                                                    num_global_views=self.num_global_views,
                                                                    num_local_views=self.num_local_views)
            lejepa_loss, a_emb = self.LeJEPA(X, channel_locations, global_views, all_views, lambd=self.lambd_lejepa,
                                                global_step=self.global_step, train_mode=True)

            loss += self.lejepa_scaling_factor * lejepa_loss

            self.log('lejepa_loss', lejepa_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            if batch_idx == 0 and self.trainer.is_global_zero and (self.current_epoch+1) % 5 == 0:
                print("Logging SIGReg 2D projection for first batch (first local view)...")
                # Pick the first view for visualization
                log_emb = a_emb[0]      # shape (B, D)
                proj2d = self.sigreg_2d_projection(log_emb, global_step=self.global_step)
                self.log_scatter_2D_SigREG(proj2d)

                print("Logging LeJEPA views for first batch (sample index 9)")
                self.log_lejepa_views(
                    X,
                    channel_names,
                    global_views,
                    all_views,
                    start_times_global,
                    start_times_local
                )

        self.log('total_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step: apply mask, normalize, compute loss and log signals.
        Args:
            batch (torch.Tensor): Input batch.
            batch_idx (int): Batch index.
        Returns:
            torch.Tensor: Loss value.
        """
        X = batch["input"]
        channel_locations = batch["channel_locations"]
        channel_names = batch.get("channel_names", None)
        mask = self.generate_mask(X.shape[0], X.shape[1], X.shape[2])

        if self.normalize:
            X = self.normalize_fct(X)

        if not self.use_lejepa_only:
            x_reconstructed, x_original, attention_scores = self.model(X, mask, channel_locations, channel_names)
            masked_loss, unmasked_loss = self.criterion(x_reconstructed, x_original, mask)
            loss = masked_loss + self.unmasked_loss_coeff * unmasked_loss

            if self.hparams.query_specialization_criterion is not None:
                query_specialization_loss = self.query_specialization_criterion(attention_scores)
                loss += query_specialization_loss
                self.log('query_specialization_loss', query_specialization_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        else:
            loss = torch.tensor(0.0, device=self.device)

        if self.use_lejepa:
            print("Computing LeJEPA loss for validation...")
            global_views, all_views, _, _= self.generate_lejepa_views(X, num_global_views=self.num_global_views,
                                                                    num_local_views=self.num_local_views)
            lejepa_loss, _ = self.LeJEPA(X, channel_locations, global_views, all_views,
                                            lambd=self.lambd_lejepa, global_step=self.global_step, train_mode=False)
            loss += self.lejepa_scaling_factor * lejepa_loss

            self.log('lejepa_val_loss', lejepa_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True) 

        # Fixed indices for logging signals
        random_indices = [6, 16, 30]

        # Log signals with mask only for the first validation batch
        if batch_idx == 0 and self.use_lejepa_only == False:
            self.log_signals_with_mask(
                x_original.float(),
                x_reconstructed.float(),
                mask,
                batch_indices=random_indices,
                indice_batch=batch_idx
            )

        return loss

    def on_validation_epoch_end(self):
        """
        Actions to perform at the end of the validation epoch.
        Currently logs t-SNE embeddings.
        """
        if not self.trainer.is_global_zero:
            print("Skipping t-SNE logging on non-zero rank.")
            return  # Only rank 0 runs t-SNE, logging, plotting

        self.visualization_loaders = {}
        if self.trainer.datamodule.tuab_loader_cfg is not None:
            print("Logging t-SNE embeddings at the end of validation epoch.")
            self.visualization_loaders['TUAB'] = self.trainer.datamodule.test_tuab_dataloader()
            self.log_tSNE_embeddings(dataset_name ="TUAB", mode="mean")
        if self.trainer.datamodule.tuar_loader_cfg is not None:
            self.visualization_loaders['TUAR'] = self.trainer.datamodule.test_tuar_dataloader()
            self.log_tSNE_embeddings(dataset_name ="TUAR", mode="mean")

    def configure_optimizers(self):
        """
        Configure optimizer and scheduler based on parameters.
        Returns:
            dict: Dictionary with optimizer and scheduler for PyTorch Lightning.
        """
        if self.hparams.optimizer.optim == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.optimizer.lr, momentum=0.9)
        elif self.hparams.optimizer.optim == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.optimizer.lr, weight_decay=0.01)
        elif self.hparams.optimizer.optim == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.optimizer.lr)
        elif self.hparams.optimizer.optim == 'LAMB':
            optimizer = torch_optim.Lamb(self.model.parameters(), lr=self.hparams.optimizer.lr)
        else:
            raise NotImplementedError("No valid optim name")

        scheduler = hydra.utils.instantiate(self.hparams.scheduler, optimizer, total_training_opt_steps=self.trainer.estimated_stepping_batches)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step_update(num_updates=self.global_step)

    def log_signals_with_mask(self, original, reconstructed, mask=None, batch_indices=None, indice_batch=None):
        """
        Log original and reconstructed signals highlighting masked regions.
        Args:
            original (torch.Tensor): Original signals.
            reconstructed (torch.Tensor): Signals reconstructed by the model.
            mask (torch.BoolTensor, optional): Applied mask.
            batch_indices (list[int], optional): Batch indices to log.
            indice_batch (int, optional): Current batch index.
        """
        patch_H, patch_W = self.patch_size
        batch_size, C, T = original.shape

        for batch_idx in batch_indices:
            original_signal = original[batch_idx]
            reconstructed_signal = reconstructed[batch_idx]

            fig, ax = plt.subplots(1, 1, figsize=(15, 6))

            # Limit visualization to the first patch_H channels
            original_signal_c2 = original_signal[:patch_H, :]
            reconstructed_signal_c2 = reconstructed_signal[:patch_H, :]

            ax.plot(original_signal_c2[0].cpu().numpy(), label='Original Channel 0', color='blue', alpha=0.7)
            ax.plot(reconstructed_signal_c2[0].cpu().numpy(), label='Reconstructed Channel 0', color='orange', alpha=0.7)

            if mask is not None:
                mask_c2 = mask[batch_idx, :patch_H, :]
                indices = []

                # Highlight masked regions with a light gray transparent band
                for i in range(patch_H):
                    for j in range(T // patch_W):
                        if mask_c2[i, j * patch_W:(j + 1) * patch_W].all():
                            ax.axvspan(j * patch_W, (j + 1) * patch_W, color='lightgray', alpha=0.1)
                            indices.append(j)

            # Remove duplicates and sort highlighted indices
            indices_array = np.array(indices)
            indices_array = np.unique(indices) 

            ax.set_title(f"Signal Reconstruction - batch_ {batch_idx}")
            ax.legend()

            # Log the figure on TensorBoard with batch and index in the title
            tag = f'Original and Reconstructed Signals with Mask (batch_0_ {batch_idx}, F1 = 0)'

            if hasattr(self.logger.experiment, 'add_figure'):  # TensorBoard
                logger = "tensorboard"
                self.logger.experiment.add_figure(tag, fig, self.current_epoch)
            elif hasattr(self.logger.experiment, 'log'):  # Wandb
                logger = "wandb"
                self.logger.experiment.log({tag: wandb.Image(fig), "epoch": self.current_epoch})


            # output as numpy image
            output_dir = os.path.join(self.logger.log_dir, 'masked_npy')
            os.makedirs(output_dir, exist_ok=True)


            # Convert figure to NumPy array
            masked_patches = []
            if mask is not None:
                mask_c2 = mask[batch_idx, :patch_H, :]
                for j in range(T // patch_W):
                    if mask_c2[:, j * patch_W:(j + 1) * patch_W].all():
                        masked_patches.append(j)

            masked_patches = np.array(masked_patches, dtype=np.int32)

            x = np.arange(T)

            original_y = original_signal_c2[0].cpu().numpy()
            reconstructed_y = reconstructed_signal_c2[0].cpu().numpy()


            # Save
            npy_filename = os.path.join(
                    output_dir,
                    f"signal_data_batch_{indice_batch}_index_{batch_idx}_epoch_{self.current_epoch}.npz"
                )

            np.savez(
                npy_filename,
                x=x,
                original_y=original_y,
                reconstructed_y=reconstructed_y,
                masked_patch_indices=masked_patches,
                epoch=self.current_epoch,
                batch_idx=batch_idx,
                channel=0,
                patch_H=patch_H,
                patch_W=patch_W,
            )

            plt.close(fig)

    def extract_test_dataset_embeddings(self, dataset_name="TUAB", mode="flatten"):
        """
        Retrieve stored embeddings from the forward hook.
        Returns:
            dict: Dictionary containing stored embeddings.
        """
        self.eval()
        all_embeddings = []
        all_labels = []

        test_loader = self.visualization_loaders[dataset_name]

        print(f"Starting embeddings extraction from test dataset with {len(test_loader)} batches.")
        print("Mode:", mode)
        device = next(self.parameters()).device

        with torch.no_grad():
            for batch in test_loader:
                x = batch["input"].to(device)
                channel_locations = batch["channel_locations"].to(device)
                channel_names = batch["channel_names"].to(device)
                labels = batch["label"]

                # fake_mask = self.generate_fake_mask(x.shape[0], x.shape[1], x.shape[2])

                if self.normalize:
                    x = self.normalize_fct(x)

                # ---- forward through encoder only ----
                z = self.model.encode(x, channel_locations)   # (B, S, d_model)

                # keep as tensor for now
                if mode == "flatten":
                    z = z.reshape(z.shape[0], -1)    # (B, S*d_model)
                elif mode == "mean":
                    z = z.mean(dim=1)                 # (B, d_model)
                all_embeddings.append(z.cpu())   # tensor on CPU

                all_labels.append(labels.cpu())

                # immediately free GPU + CPU references
                del x, channel_locations, z, labels

        torch.cuda.empty_cache()

        # ---- concatenate before conversion ----
        all_embeddings = torch.cat(all_embeddings, dim=0)  # tensor (N, S*d_model)
        all_labels = torch.cat(all_labels, dim=0)          # tensor (N,)

        emb_np = all_embeddings.numpy()
        lab_np = all_labels.numpy()

        print(f"Extracted embeddings from test dataset for {len(test_loader)} batches.")
        print(f"{len(all_embeddings)} embeddings collected.")
        print(f"All embeddings shape: {all_embeddings.shape}, All labels shape: {all_labels.shape}")

        # free torch tensors before returning numpy
        del all_embeddings, all_labels
        gc.collect()

        # ---- convert ONCE at the end ----
        return emb_np, lab_np

    def log_tSNE_embeddings(self, dataset_name="TUAB", mode="flatten"):
        """
        Log t-SNE embeddings as a scatter plot.
        Args:
            embeddings_2d (np.ndarray): 2D embeddings of shape (N, 2).
            labels (np.ndarray): Labels corresponding to each embedding of shape (N,).
            title_suffix (str, optional): Suffix to add to the plot title.
        """
        print("Collecting embeddings...")
        encoder_embeddings, labels = self.extract_test_dataset_embeddings(dataset_name=dataset_name,
                                                                          mode=mode)

        print("Running t-SNE...")
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=30,
            learning_rate='auto'
        )
        embed_2d = tsne.fit_transform(encoder_embeddings) 
        print("t-SNE projection completed.")

        if dataset_name == "TUAB":
            color_mapping = {0: "#0004fd", 1: "#ff0303"}
            label_mapping = {0: "Normal signal", 1: "Abnormal signal"}
        elif dataset_name == "TUAR":
            color_mapping = {
                0: "#93a7b3ff",  # blue
                1: "#2D8535",  # orange
                2: "#d87426",  # green
                3: "#3272D8",  # red
                4: "#f452b3",  # purple
                5: "#653C8B"   # brown
            }
            label_mapping = {0: "Normal", 1: "Chewing", 
                    2: "Electrode artifact", 3: "Eye movement",
                    4: "Muscle movement", 5: "Shivering"}
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        # labels = np.array(labels).astype(int).squeeze()
        print(f"Labels unique values and counts: {np.unique(labels, return_counts=True)}")
        print(f"Embeddings shape: {embed_2d.shape}, Labels shape: {labels.shape}")

        colors = [color_mapping[label] for label in labels]
        fig = plt.figure(figsize=(8, 6), dpi=150)
        scatter = plt.scatter(
            embed_2d[:, 0],
            embed_2d[:, 1],
            c=colors,
            alpha=1.,
            edgecolors='none',
            s=10
        )

        epoch_num = f"Epoch {self.current_epoch}"

        legend_elements = [
            plt.Line2D([0], [0],
                    marker='o',
                    linestyle='None',
                    markerfacecolor=color_mapping[lbl],
                    markeredgecolor='none',
                    markersize=6,
                    label=label_mapping[lbl])
            for lbl in label_mapping
        ]

        plt.legend(handles=legend_elements, loc="upper right")
        plt.title(f"t-SNE Projection for {dataset_name} ({epoch_num})")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")

        # ---------- Logging ----------
        if hasattr(self.logger.experiment, 'log'):  # wandb
            self.logger.experiment.log({
                f"tSNE/{dataset_name}": wandb.Image(fig),
                "epoch": self.current_epoch
            })
        elif hasattr(self.logger.experiment, 'add_figure'):  # TensorBoard
            tag = f"tSNE/{dataset_name}"
            self.logger.experiment.add_figure(tag, fig,
                                              global_step=self.current_epoch)

        # output as npy file to output directory
        output_dir = os.path.join(self.logger.log_dir, 'tSNE_npy')
        os.makedirs(output_dir, exist_ok=True)

        npz_filename = os.path.join(
            output_dir,
            f"tSNE_{dataset_name}_epoch_{self.current_epoch}.npz"
        )

        np.savez(
            npz_filename,
            embeddings_2d=embed_2d.astype(np.float32),
            labels=labels.astype(np.int32),
            dataset_name=dataset_name,
            epoch=self.current_epoch,
            mode=mode,
        )

        print(f"Saved t-SNE data to {npz_filename}")


        plt.close(fig)

        # hard cleanup
        del embed_2d
        del encoder_embeddings
        del labels
        del tsne

        gc.collect()
        torch.cuda.empty_cache()