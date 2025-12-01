import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from .unet import UNet

class FBPUNet(pl.LightningModule):

    def __init__(self, depth=6, base_filters=32, dim=3, norm='instance'):
        super().__init__()
        self.unet = UNet(1, 1, depth, base_filters, dim, norm=norm)

        self.save_hyperparameters()

    def forward(self, x):
        return self.unet(x)
    
    def training_step(self, batch, batch_idx):
        fbp = batch['fbp']
        volume = batch['volume']
        output = self.unet(fbp)
        loss = F.mse_loss(output, volume)
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        fbp = batch['fbp']
        volume = batch['volume']
        output = self.unet(fbp)
        loss = F.mse_loss(output, volume)
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        return loss
    
    def configure_callbacks(self):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            filename='fbp_unet-{epoch:02d}-{val_loss:.4f}',
            mode='min',
            save_last=True,
        )
        return [checkpoint_callback]
