import logging
import warnings
import numpy as np
from scipy.stats import norm
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.modules import UniformFlow
from src.utils.utils import strip_outliers

# logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.basicConfig()
logger = logging.getLogger(__name__)

EPS = 1e-6
class CCNF(LightningModule):
    def __init__(self, dx, dy, dz, n_components, hidden_sizes, lr, weight_decay, verbose):
        super().__init__()
        self.save_hyperparameters()

        self.x_cnf = UniformFlow(d=dx, dz=dz, n_components=n_components, dh=hidden_sizes)
        self.y_cnf = UniformFlow(d=dy, dz=dz, n_components=n_components, dh=hidden_sizes)

    def loss(self, X, Y, Z):
        ex, log_px = self.x_cnf(X, Z)
        ey, log_py = self.y_cnf(Y, Z)

        loss = -torch.mean(log_px + log_py)
        return loss

    def training_step(self, batch, batch_idx):
        X, Y, Z = batch
        loss = self.loss(X, Y, Z)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y, Z = batch
        loss = self.loss(X, Y, Z)

        return loss
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log('val_loss', avg_loss)
        
    def transform(self, X, Y, Z):
        self.eval()
        ex, log_px = self.x_cnf(X, Z)
        ey, log_py = self.y_cnf(Y, Z)

        ex = ex.detach().cpu().numpy()
        ey = ey.detach().cpu().numpy()

        return ex, ey

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, factor=.1, patience=10, verbose=self.hparams.verbose)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    def fit(self, X, Y, Z, max_epochs, verbose, gpus=None, callbacks=None):
        gpus = gpus or 0
        callbacks = callbacks or []
        N = X.shape[0]
        train_size = int(N * 0.7)
        valid_size = N - train_size
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            train, valid = random_split(TensorDataset(X, Y, Z), lengths=[train_size, valid_size])
            train_dataloader = DataLoader(train, batch_size=64, shuffle=True)
            val_dataloader = DataLoader(valid, batch_size=64)
            early_stopping = EarlyStopping(
                mode='min',
                monitor='val_loss',
                patience=10,
                verbose=verbose
            )
            callbacks.append(early_stopping)
            trainer = Trainer(
                accelerator="gpu" if torch.cuda.device_count() else "cpu",
                gpus=gpus if torch.cuda.device_count() else 0,
                max_epochs=max_epochs,
                logger=verbose,
                enable_checkpointing=verbose,
                enable_progress_bar=verbose,
                enable_model_summary=verbose,
                deterministic=True,
                callbacks=callbacks,
                detect_anomaly=False,
                gradient_clip_val=1,
                gradient_clip_algorithm="value"
            )
            trainer.fit(model=self, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
            logs, = trainer.validate(model=self, dataloaders=val_dataloader, verbose=verbose)
        
        return logs

def DINE(X, Y, Z, normalize=True, n_components=32, hidden_sizes=4, lr=5e-3, weight_decay=5e-5, max_epochs=100, random_state=0, gpus=0, return_latents=False, verbose=False, **kwargs):
    logger.setLevel(logging.DEBUG if verbose else logging.ERROR)
    if random_state is not None:
        torch.random.fork_rng(enabled=True)
        torch.random.manual_seed(random_state)

    N = X.shape[0]
    if normalize:
        X, Y, Z = map(lambda x: (x - np.mean(x)) / np.std(x), (X, Y, Z))
    X, Y, Z = map(lambda x: torch.tensor(x).float().view(N, -1), (X, Y, Z))

    model = CCNF(dx=X.shape[1], dy=Y.shape[1], dz=Z.shape[1], n_components=n_components, hidden_sizes=hidden_sizes, lr=lr, weight_decay=weight_decay, verbose=verbose)
    model.fit(X, Y, Z, max_epochs=max_epochs, verbose=verbose, gpus=gpus)
    
    e_x, e_y = model.transform(X, Y, Z)
    e_x, e_y = map(lambda x: np.clip(x, EPS, 1 - EPS), (e_x, e_y))
    e_x, e_y = map(norm.ppf, (e_x, e_y))
    
    cov_x, cov_y = map(np.cov, (e_x.T, e_y.T))
    cov_x = cov_x.reshape(e_x.shape[1], e_x.shape[1])
    cov_y = cov_y.reshape(e_y.shape[1], e_y.shape[1])
    cov_all = np.cov(np.column_stack((e_x, e_y)).T)
    mi = 0.5 * (np.log(np.linalg.det(cov_x)) + np.log(np.linalg.det(cov_y)) - np.log(np.linalg.det(cov_all)))

    assert mi == mi

    if return_latents:
        return e_x, e_y, mi

    return mi