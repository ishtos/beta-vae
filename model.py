import torch
import torch.nn as nn
import torch.nn.init as init


class View(nn.Module):

    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.view(self.shape)


class BetaVAE(nn.Module):

    def __init__(self, in_channels=3, n_channels=32, n_latent=10):
        super(BetaVAE, self).__init__()
        self.n_latent = n_latent

        ## define encoder
        self.encoder = nn.Sequential(
            self._build_conv_block(in_channels, n_channels),
            self._build_conv_block(n_channels, n_channels),
            self._build_conv_block(n_channels, n_channels*2),
            self._build_conv_block(n_channels*2, n_channels*2),
            self._build_conv_block(n_channels*2, n_channels*8, stride=1),
            View((-1, n_channels*8)),
            nn.Linear(n_channels*8, n_latent*2)
        )

        ## define decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_channels*8),
            View((-1, n_channels*8, 1, 1)),
            self._build_dconv_block(n_channels*8, n_channels*2, stride=1),
            self._build_dconv_block(n_channels*2, n_channels*2),
            self._build_dconv_block(n_channels*2, n_channels),
            self._build_dconv_block(n_channels, n_channels),
            self._build_dconv_block(n_channels, in_channels),
            nn.Sigmoid()
        )

        ## initialize weights for Conv2d, ConvTranspose2d, Linear 
        self.init_weights()

    def _build_conv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        model = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*model)
    
    def _build_dconv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        model = [
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding)
        ]
        return nn.Sequential(*model)

    def init_weights(self):
        for block in self._modules:
            for module in self._modules[block]: 
                if isinstance(module, nn.Sequential):   
                    if isinstance(module[0], nn.Conv2d):
                        init.kaiming_normal_(module[0].weight)
                    elif isinstance(module[1], nn.ConvTranspose2d):
                        init.kaiming_normal_(module[1].weight)
                elif isinstance(module, nn.Linear):
                    init.kaiming_normal_(module.weight)

    def _encoder(self, x):
        distributions = self.encoder(x)
        return distributions[:, :self.n_latent], distributions[:, self.n_latent:]   
    
    def _decoder(self, x):
        return self.decoder(x)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self._encoder(x)
        z = self.reparameterize(mu, logvar)
        return self._decoder(z), mu, logvar




