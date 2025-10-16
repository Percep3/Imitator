from tracemalloc import start
from tqdm import tqdm
import os
import typing as t

from accelerate import Accelerator
import torch
import random

from src.mslm.lib.clean_attn import clean_and_mean_attn, plot_attn_with_metrics, _fit_tokens_mask_to_Lq, _fit_frames_mask_to_Lk

torch.manual_seed(23)
random.seed(23)

from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

from src.mslm.utils.early_stopping import EarlyStopping
from src.mslm.checkpoint.manager import CheckpointManager
from src.mslm.metrics import AlignmentMetrics, MetricsLogger
# from src.mslm.training import imitator_loss
from src.mslm.training.loss_msepcossim import imitator_loss
import nvtx
from datetime import datetime

def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x: [B, T, D]
    mask: [B, T] (1/True = válido)
    """
    m = mask.to(dtype=x.dtype).unsqueeze(-1)     # [B,T,1]
    num = (x * m).sum(dim=1)                     # [B,D]
    den = m.sum(dim=1).clamp_min(1e-8)           # [B,1]
    return num / den

class Trainer:
    def __init__(self, model, train_loader, val_loader, learning_rate, save_tb_model=True, **kwargs):
        #dynamo_plugin = TorchDynamoPlugin(
        #    backend="inductor",  # Options: "inductor", "aot_eager", "aot_nvfuser", etc.
        #    mode="default",      # Options: "default", "reduce-overhead", "max-autotune"
        #    dynamic=True
        #)
        
        # Metrics
        self.align_meter = AlignmentMetrics(device=kwargs.get("device", "gpu" if torch.cuda.is_available() else "cpu"))
        self.metrics_logger = MetricsLogger(save_dir="../outputs/reports/metrics")
        
        #Accelerator module
        self.accelerator = Accelerator(mixed_precision="bf16")#, dynamo_plugin=dynamo_plugin)
        self.device = self.accelerator.device

        #Hyperparameters
        self.epochs = kwargs.get("epochs", 100)
        self.learning_rate = learning_rate

        #Loggers
        self.log_interval = kwargs.get("log_interval", 5)
        self.save_tb_model = save_tb_model

        version = kwargs.get("model_version", 1)
        checkpoint = kwargs.get("checkpoint", 1)

        self.writer = SummaryWriter(f"../outputs/reports/{version}/{checkpoint}/{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}")
        self.graph_added = False
        
        #Save and checkpoint
        self.checkpoint_interval = kwargs.get("checkpoint_interval", 5)
        self.ckpt_mgr = CheckpointManager(
            kwargs.get("model_dir", "../outputs/checkpoints"),
            version,
            checkpoint,
        )

        #Loss Function
        if kwargs.get("compile", True):
            self.criterion = torch.compile(
                imitator_loss,
                backend="inductor",
                mode="default",
                dynamic=True
            )            
        else:
            self.criterion = imitator_loss

        #Model
        self.model = model
        self.load_previous_model = kwargs.get("load_previous_model", False)
                
        #Dataloaders
        self.train_loader = self.accelerator.prepare_data_loader(train_loader)
        self.val_loader = self.accelerator.prepare_data_loader(val_loader)

        #Stopper
        self.early_stopping = EarlyStopping(patience=10, threshold=0.00005)

        #Optimizer
        self.optimizer = None
        self.scheduler = None

        #Batch Sampling
        self.batch_size = kwargs.get("batch_size", 5)
        self.batch_sampling = kwargs.get("batch_sampling", True)
        if self.batch_sampling:
            self.sub_batch = kwargs.get("batch_sample", 4)

        #Options 
        self.prof = False
        self.distributed = None
        
        self.grad_clip = kwargs.get("grad_clip", 0.1)
        self.weight_decay = kwargs.get("weight_decay", 0.05)


    def prepare_trainer(self):
        """Prepara todo lo necesario para el entrenamiento."""
        (self.model,
        self.criterion,
        self.optimizer,
        self.scheduler,
        ) = self.accelerator.prepare(
            self.model, self.criterion, self.optimizer, self.scheduler,
        )
        
    @nvtx.annotate("Training Section", color="green")
    def train(self, prof = False, load=False):
        """Entrena el modelo Imitator.
        returns:
            train_loss: float, loss de entrenamiento
            val_loss: float, loss de validación
        """
        print("LR:", self.learning_rate)
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
            foreach=True
        )
        
        def linear_warmup_cosine_decay(current_step, warmup_steps, total_steps):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 0.5 * (1.0 + torch.cos(
                torch.tensor((current_step - warmup_steps) / (total_steps - warmup_steps) * 3.1415926535))
            ).item()

        warmup_steps = 5 * len(self.train_loader)  # p.ej. 5 epochs de warm-up
        total_steps = self.epochs * len(self.train_loader)

        lr_lambda = lambda step: linear_warmup_cosine_decay(step, warmup_steps, total_steps)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        if self.load_previous_model: 
            self.ckpt_mgr.load_checkpoint(self.model, self.optimizer, self.scheduler)

        self.prepare_trainer()
        self.prof = prof

        for epoch in tqdm(range(self.epochs), desc="Entrenando", colour="green"):
            train_loss = self._train_epoch(epoch)
            val_loss = self._val(epoch)

            if epoch == 1:
                self.ckpt_mgr.save_checkpoint(self.model, epoch, self.optimizer, self.scheduler)
            elif epoch == self.epochs - 1:
                self.ckpt_mgr.save_checkpoint(self.model, epoch, self.optimizer, self.scheduler)
            elif (epoch % self.checkpoint_interval == 0 and epoch != 0) :
                self.ckpt_mgr.save_checkpoint(self.model, epoch, self.optimizer, self.scheduler)
            elif self.early_stopping.stop:
                self.ckpt_mgr.save_checkpoint(self.model, epoch, self.optimizer, self.scheduler)
            
            if self.scheduler is not None:
                self.scheduler.step()

            if self.early_stopping.stop:
                break

        return train_loss, val_loss
    
    @nvtx.annotate("Train: Train Epoch", color="green")
    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        mse_loss = 0
        cossim_loss = 0
        for keypoint, frames_padding_mask, embedding, mask_embedding, _, _ in self.train_loader:
            if self.save_tb_model and epoch == 1 and not getattr(self, "graph_added", False):
                print("Saving graph")
                self.writer.add_graph(self.model, (keypoint, frames_padding_mask))
                self.graph_added = True           
            
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad(set_to_none=True)        
                train_loss, mse, cossim = self._train_batch(keypoint, frames_padding_mask, embedding, mask_embedding)

            if self.distributed is not None:
                loss_tensor = train_loss.to(self.device)
                self.distributed.all_reduce(loss_tensor, op=self.distributed.ReduceOp.SUM)
                train_loss = (loss_tensor) / self.distributed.get_world_size()
                if self.distributed.get_rank() == 0:
                    print(f"World-avg train loss: {train_loss:.4f}")
            else:
                total_loss += train_loss
                mse_loss += mse
                cossim_loss += cossim
                
        final_train_loss = total_loss.item()/len(self.train_loader)
        final_train_loss_mse = mse_loss.item()/len(self.train_loader)
        final_train_loss_cossim = cossim_loss.item()/len(self.train_loader)

        self.writer.add_scalar("Loss/train", final_train_loss, epoch)
        self.writer.add_scalar("Loss/train_mse", final_train_loss_mse, epoch)
        self.writer.add_scalar("Loss/train_cosim", final_train_loss_cossim, epoch)

        if epoch % self.log_interval == 0:
            tqdm.write(f"\nEpoch: {epoch}.\n Train loss: {final_train_loss} MSE: {final_train_loss_mse} Cossim: {final_train_loss_cossim}")

        return total_loss

    def _forward_loss(self, keypoint, frames_padding_mask, embedding, mask_embedding, return_output=False) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, t.Optional[torch.Tensor]]:
        with self.accelerator.autocast():
            output, _ = self.model(keypoint, frames_padding_mask)
            token_len = output.size(1)

            tgt = embedding
            emb_pad = mask_embedding.bool()
            frm_pad = frames_padding_mask.bool()

            if tgt.size(1) < token_len:
                pad_tokens = token_len - tgt.size(1)
                pad_values = tgt.new_zeros(tgt.size(0), pad_tokens, tgt.size(2))
                tgt = torch.cat([tgt, pad_values], dim=1)
                pad_mask_ext = torch.ones(
                    emb_pad.size(0),
                    pad_tokens,
                    dtype=emb_pad.dtype,
                    device=emb_pad.device,
                )
                emb_pad = torch.cat([emb_pad, pad_mask_ext], dim=1)
            elif tgt.size(1) > token_len:
                tgt = tgt[:, :token_len]
                emb_pad = emb_pad[:, :token_len]

            tokens_pad = _fit_tokens_mask_to_Lq(emb_pad, token_len)
            frames_pad = _fit_frames_mask_to_Lk(frm_pad, token_len)
            pad_mask = torch.logical_or(tokens_pad, frames_pad)

            valid = (~pad_mask).unsqueeze(-1).to(dtype=output.dtype)
            masked_output = output * valid
            masked_target = tgt * valid

            loss, mse, cossim = self.criterion(masked_output, masked_target, pad_mask)
        if return_output:
            return loss, mse, cossim, masked_output
        return loss, mse, cossim, None

    @nvtx.annotate("Train: Train Batch", color="green")
    def _train_batch(self, keypoint, frames_padding_mask, embedding, mask_embedding):
        keypoint = keypoint.to(self.device, non_blocking=True)
        embedding = embedding.to(self.device, non_blocking=True)

        frames_padding_mask = frames_padding_mask.bool().to(self.device, non_blocking=True)
        mask_embedding = mask_embedding.bool().to(self.device, non_blocking=True)

        if keypoint.dim() == 4 and keypoint.size(2) > keypoint.size(3):
            keypoint = keypoint.permute(0, 1, 3, 2).contiguous()

        if frames_padding_mask.size(1) < keypoint.size(1):
            pad = torch.ones(
                frames_padding_mask.size(0),
                keypoint.size(1) - frames_padding_mask.size(1),
                dtype=frames_padding_mask.dtype,
                device=self.device,
            )
            frames_padding_mask = torch.cat([frames_padding_mask, pad], dim=1)
        elif frames_padding_mask.size(1) > keypoint.size(1):
            frames_padding_mask = frames_padding_mask[..., : keypoint.size(1)]

        if mask_embedding.size(1) < embedding.size(1):
            pad = torch.ones(
                mask_embedding.size(0),
                embedding.size(1) - mask_embedding.size(1),
                dtype=mask_embedding.dtype,
                device=self.device,
            )
            mask_embedding = torch.cat([mask_embedding, pad], dim=1)
        elif mask_embedding.size(1) > embedding.size(1):
            mask_embedding = mask_embedding[..., : embedding.size(1)]

        batch_loss = 0.0
        batch_mse, batch_cossim = 0.0, 0.0 

        batch_size = keypoint.size(0)
        start = 0
        end = keypoint.size(0)
        if self.batch_sampling:
            n_sub_batch = (batch_size + self.sub_batch - 1) // self.sub_batch

        with nvtx.annotate("Sub_batch", color="blue"):
            with torch.autograd.set_detect_anomaly(True):
                for i in range(n_sub_batch):
                    if self.batch_sampling:
                        start = i * self.sub_batch
                        end = min(start + self.sub_batch, batch_size)
                        with nvtx.annotate("Forward Pass", color="blue"):
                            loss, mse, cossim, _= self._forward_loss(keypoint[start:end], 
                                                        frames_padding_mask[start:end], 
                                                        embedding[start:end], 
                                                        mask_embedding[start:end])
                        if self.batch_sampling:
                            loss /= n_sub_batch
                            mse /= n_sub_batch
                            cossim /= n_sub_batch

                        with nvtx.annotate("Backward Pass", color="blue"):
                            torch.autograd.set_detect_anomaly(True)
                            self.accelerator.backward(loss)
                        batch_loss += loss.detach()
                        batch_mse += mse.detach()
                        batch_cossim += cossim.detach()

            with nvtx.annotate("Step", color="blue"):
                self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()

        return batch_loss, batch_mse, batch_cossim

    @nvtx.annotate("Validation Section", color="green")
    def _val(self, epoch):
        self.model.eval()
        val_loss=0
        mse_loss = 0
        cossim_loss = 0
        for keypoint, frames_padding_mask, embedding, mask_embedding, _, _ in self.val_loader:        
            loss, mse, cossim = self._val_batch(epoch, keypoint, frames_padding_mask, embedding, mask_embedding)
            if self.distributed is not None:
                loss_tensor = loss.to(self.device)
                self.distributed.all_reduce(loss_tensor, op=self.distributed.ReduceOp.SUM)
                val_loss = (loss_tensor) / self.distributed.get_world_size()
                if self.distributed.get_rank() == 0:
                    print(f"World-avg val loss: {loss:.4f}")
            else:
                val_loss += loss
                mse_loss += mse
                cossim_loss += cossim

        final_val_loss = val_loss.item() / len(self.val_loader)
        final_mse_loss = mse_loss.item() / len(self.val_loader)
        final_cossim_loss = cossim_loss.item() / len(self.val_loader)
        self.writer.add_scalar("Loss/val", final_val_loss, epoch)
        self.writer.add_scalar("Loss/val_mse", final_mse_loss, epoch)
        self.writer.add_scalar("Loss/val_cossim", final_cossim_loss, epoch)

        if epoch % self.log_interval == 0:
            tqdm.write(f"Validation loss: {final_val_loss} MSE: {final_mse_loss} Cossim: {final_cossim_loss}")
        
        try:
            if hasattr(self, "align_meter") and hasattr(self, "metrics_logger"):
                summary = self.align_meter.summary()  # dict de métricas agregadas de esta época
                if summary:
                    # log a TensorBoard bajo el prefijo "Align/*"
                    self.metrics_logger.log_epoch(self.writer, epoch, summary, prefix="Align")

                    # log corto también por consola cada log_interval
                    if epoch % self.log_interval == 0:
                        # selecciona algunas claves típicas si existen
                        keys_prefer = [
                            "alignment_mse", "cka_xy", "procrustes_err",
                            "R@1_x2y", "R@1_y2x", "R@5_x2y", "R@5_y2x",
                            "attn_entropy", "attn_monotonicity"
                        ]
                        summary_str = " ".join(
                            f"{k}:{summary[k]:.4f}" for k in keys_prefer if k in summary
                        )
                        if summary_str:
                            tqdm.write(f"[Align] epoch {epoch}  {summary_str}")
                            
                    if (epoch == 0) or (epoch % 5 == 0) or (epoch == self.epochs - 1):
                        self.metrics_logger.save_curves()
                # reset para la próxima época
                self.align_meter.reset()
        except Exception as e:
            tqdm.write(f"[Align][WARN] métricas no registradas por: {repr(e)}")
            
        self.early_stopping(final_val_loss)
        return final_val_loss

    @nvtx.annotate("Val: Validate Batch", color="green")
    def _val_batch(
        self,
        epoch: int,
        keypoint,
        frames_padding_mask,
        embedding,
        mask_embedding
    ) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Validación con sub-batching, forward único (output + attn),
        logging de curvas de atención y acumulación de métricas de alineación.
        Retorna: (batch_loss, batch_mse, batch_cossim) como tensores acumulados.
        """

        def masked_mean(x: torch.Tensor, mask_valid: torch.Tensor) -> torch.Tensor:
            """
            x: [B, T, D]
            mask_valid: [T] o [B, T] con True=VÁLIDO.
            devuelve: [B, D]
            """
            if mask_valid.dim() == 1:
                mask_valid = mask_valid.unsqueeze(0).to(x.device)       # [1, T]
            else:
                mask_valid = mask_valid.to(x.device)

            B, T, D = x.shape
            if mask_valid.size(0) == 1 and B > 1:
                mask_valid = mask_valid.expand(B, -1)                   # [B, T]
            assert mask_valid.size(0) == B and mask_valid.size(1) == T, \
                f"mask_valid {mask_valid.shape} no coincide con x {x.shape}"

            m = mask_valid.to(dtype=x.dtype).unsqueeze(-1)              # [B, T, 1]
            num = (x * m).sum(dim=1)                                    # [B, D]
            den = m.sum(dim=1).clamp_min(1e-8)                          # [B, 1]
            return num / den

        self.model.eval()
        batch_loss = torch.tensor(0.0, device=self.device)
        batch_mse = torch.tensor(0.0, device=self.device)
        batch_cossim = torch.tensor(0.0, device=self.device)

        batch_size = keypoint.size(0)
        n_sub_batch = (batch_size + self.sub_batch - 1) // self.sub_batch if self.batch_sampling else 1

        with nvtx.annotate("Val: Forward + Loss", color="blue"):
            for i in range(n_sub_batch):
                if self.batch_sampling:
                    start = i * self.sub_batch
                    end = min(start + self.sub_batch, batch_size)
                else:
                    start, end = 0, batch_size

                # Cortes del sub-batch
                kp_sb   = keypoint[start:end]
                fpm_sb  = frames_padding_mask[start:end]   # usualmente True=PAD
                tgt_sb  = embedding[start:end]             # [B,Tt,Dt] o [B,Dt]
                tmask_sb= mask_embedding[start:end]        # [B,Tt]  True/1=VALID (ajusta si hace falta)

                # Forward único para obtener output y atención
                with self.accelerator.autocast():
                    # Asegúrate que tu modelo soporte return_attn=True devolviendo (out, attn_w)
                    # attn_w: [B, H, Lq, Lk]; aquí queremos tokens->frames
                    output_sb, attn_w = self.model(kp_sb, fpm_sb, return_attn=True)

                    # Pérdidas (tu criterion ya consume [pred, target, mask_tokens])
                    loss, mse, cossim = self.criterion(output_sb, tgt_sb, tmask_sb)

                # Normaliza por número de sub-batches si corresponde
                if self.batch_sampling:
                    loss   = loss   / n_sub_batch
                    mse    = mse    / n_sub_batch
                    cossim = cossim / n_sub_batch

                # Backprop NO aquí (esto es validación), solo acumular
                batch_loss   += loss.detach()
                batch_mse    += mse.detach()
                batch_cossim += cossim.detach()

                # --- Atención: limpiar, promediar cabezas y plotear cada 5 épocas (o la 0 y la final)
                if i == 0 and (epoch == 0 or epoch % 5 == 0 or epoch == self.epochs - 1):
                    with torch.no_grad():
                        # attn_w: [B, H, Lq, Lk] -> limpieza y media por cabezas: [B, Lq, Lk]
                        attn_w_cpu     = attn_w.detach().cpu()
                        frames_mask_cpu= fpm_sb.cpu()          # [B, Lk] True=PAD en tu convención
                        tokens_mask_cpu= tmask_sb.cpu().bool() # [B, Lq] True=PAD (asegurar bool)
                        b = 0

                        A_mean = clean_and_mean_attn(attn_w_cpu, frames_mask_cpu[b], tokens_mask_cpu[b])  # [B,Lq,Lk]

                        # Plot de un ejemplo
                        plot_attn_with_metrics(
                            A_mean_b=A_mean[b],                         # [Lq,Lk]
                            frames_pad_mask_b=frames_mask_cpu[b],
                            tokens_pad_mask_b=(~tokens_mask_cpu[b]),
                            epoch=epoch,
                            title_prefix=f"Cross-attention (tokens→frames)",
                            savepath=f"../outputs/reports/attn/attn_epoch{epoch}.png"
                        )

                    # --- Métricas de alineación (si tienes el acumulador configurado)
                    with torch.no_grad():
                        # Convención de máscaras para métricas:
                        # frames_padding_mask viene como True=PAD → convertir a True=VALID
                        frames_valid = (~fpm_sb).bool()           # [B, T_frames]
                        tokens_valid = tmask_sb.bool()            # [B, T_tokens] True=VALID

                        # Pool a [B,D] (si target ya es [B,D], omite el pooling de target)
                        if output_sb.dim() == 3:
                            Lq = output_sb.size(1)
                            # True=PAD para queries del sub-batch:
                            q_pad = _fit_tokens_mask_to_Lq(tmask_sb.bool(), Lq=Lq)  # [B, Lq] True=PAD
                            tokens_valid_pred = (~q_pad).bool()                         # [B, Lq] True=VALID
                            pred_vec = masked_mean(output_sb, tokens_valid_pred)
                        else:
                            pred_vec = output_sb

                        if tgt_sb.dim() == 3:
                            Tt = tgt_sb.size(1)
                            q_pad_tgt = _fit_tokens_mask_to_Lq(tmask_sb.bool(), Lq=Tt)  # [B, Tt] True=PAD
                            tokens_valid_tgt = (~q_pad_tgt).bool()                          # [B, Tt] True=VALID
                            target_vec = masked_mean(tgt_sb, tokens_valid_tgt)
                        else:
                            target_vec = tgt_sb

                        # 3) Normalizar orientación de atención a [B, H, Ty, Tx] y luego promediar cabezas
                        A_mean_gpu = None
                        try:
                            Aw = attn_w  # [B, H, ?, ?]
                            B_, H_, A0, A1 = Aw.shape
                            Ty = tokens_valid_pred.shape[1]
                            Tx = frames_valid.shape[1]

                            # Queremos Aw: [B,H,Ty,Tx]
                            if (A0, A1) == (Ty, Tx):
                                Aw_norm = Aw
                            elif (A0, A1) == (Tx, Ty):
                                Aw_norm = Aw.transpose(-1, -2)  # [B,H,Ty,Tx]
                            else:
                                # Si no calza exactamente, elige la permuta que case con Ty y Tx
                                if A0 == Ty and A1 != Tx:
                                    # último dim no coincide -> trata de alinear al tamaño de frames
                                    Aw_norm = Aw[..., :, :Tx]  # o lanza excepción si prefieres
                                elif A1 == Tx and A0 != Ty:
                                    Aw_norm = Aw[..., :Ty, :]
                                else:
                                    # Como fallback, no uses attn métricas
                                    Aw_norm = None

                            if Aw_norm is not None:
                                A_mean_gpu = Aw_norm.mean(dim=1)    # [B, Ty, Tx]
                                # Opcional: re-normaliza por si hubo recortes
                                A_mean_gpu = torch.softmax(A_mean_gpu, dim=-1)
                        except Exception:
                            A_mean_gpu = None  # si algo falla, seguimos sin métricas de attn


                        self.align_meter.accumulate(
                            pred=pred_vec.detach(),
                            target=target_vec.detach(),
                            token_mask=tokens_valid_pred,   # type: ignore
                            frame_mask=frames_valid,
                            attn=A_mean_gpu,          # opcional
                            pairs=None,
                            compute_cka=True,
                            compute_procrustes=True,
                        )

        return batch_loss, batch_mse, batch_cossim
