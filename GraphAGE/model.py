import torch
import torch.nn as nn
from functools import partial
from loss import sce_loss
from GAT import GAT


class AEModel(nn.Module):
    def __init__(self, args, in_dim):
        super().__init__()
        self.args = args
        self.in_dim = in_dim

        self.num_layers = args.num_layers
        self.num_hidden = args.num_hidden
        self.num_heads = args.num_heads
        self.num_out_heads = args.num_out_heads

        self.criterion = partial(sce_loss, alpha=2)

        enc_num_hidden = self.num_hidden // self.num_heads
        dec_in_dim = self.num_hidden
        dec_num_hidden = self.num_hidden // self.num_out_heads

        self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        self.encoder = GAT(
            self.args,
            self.in_dim,
            enc_num_hidden,
            enc_num_hidden,
            self.num_layers,
            self.num_heads,
            encoding=True
        )

        self.decoder = GAT(
            self.args,
            dec_in_dim,
            self.in_dim,
            dec_num_hidden,
            num_layers=1,
            num_out_heads=1,
            encoding=False
        )

    def forward(self, g, x):
        enc_rep, _all_hidden = self.encoder(g, x, return_hidden=True)
        rep = self.encoder_to_decoder(enc_rep)
        recon = self.decoder(g, rep)
        loss = self.criterion(recon, x)
        loss_item = {"loss": loss.item()}
        return loss, loss_item, enc_rep, recon

    @torch.no_grad()
    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep
