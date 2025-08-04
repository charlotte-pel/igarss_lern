import torch as th
import torch.nn as nn
import geoopt
from utils import (
    _axat, 
    _atxa, 
    _mvmt, 
)

class ResidualBlock(nn.Module):
    def __init__(self, dim, manifold_SPD, manifold_Stiefel, proj=False):
        super(ResidualBlock, self).__init__()

        # Manifold operations
        self.manifold_SPD = manifold_SPD
        self.manifold_Stiefel = manifold_Stiefel
        
        # Exponential mapping
        self.proj = proj

        # Each ResidualBlock has its own projection matrix P1
        P = th.randn((1, dim, dim), dtype=th.float64)
        P = th.svd(P)[0]
        self.register_parameter("P1", geoopt.ManifoldParameter(P, self.manifold_Stiefel))
        
        # Spectrum mapping
        self.spectrum_map = nn.Sequential(
            nn.Conv1d(1, 10, 5, padding="same").double(),
            nn.LeakyReLU(),
            nn.BatchNorm1d(10).double(),
            nn.Conv1d(10, 5, 3, padding="same").double(),
            nn.LeakyReLU(),
            nn.BatchNorm1d(5).double(),
            nn.Conv1d(5, 1, 3, padding="same").double(),
        )

    def forward(self, x):
        # Perform the residual operation: transformation + residual addition
        evecs, eigs, _ = th.svd(x)
        f_eigs = self.spectrum_map(eigs)
        v1 = _mvmt(self.P1, f_eigs, self.P1)
        v1 = self.manifold_SPD.proju(x, v1) # projects v1 onto the tangent space of x
        eigs = th.clamp(eigs, 1e-8, 1e8) # equivalent to LogEig
        log_x = _mvmt(evecs, th.log(eigs), evecs)
        x = log_x + v1
        if self.proj:
            return self.manifold_SPD.projx(x)
        else:
            return x
    
class LogEucRResNet(nn.Module):
    def __init__(self, inputdim, dim1, classes, n_blocks=1, embed_only=False, classifier="linear"):
        """
        Single-modality Log-Euclidean Residual Network.

        Args:
            inputdim (int): Input dimension.
            dim1 (int): Dimension after first bi-map.
            classes (int): Number of output classes.
            embed_only (bool): If True, returns embeddings only.
            classifier (str): Type of classifier ("linear").
        """
        super().__init__()
        self.inputdim = inputdim
        self.dim1 = dim1
        self.classes = classes
        self.embed_only = embed_only
        self.classifier = classifier
        self.n_blocks = n_blocks

        # Manifolds
        self.manifold_Stiefel = geoopt.Stiefel()
        self.manifold_SPD = geoopt.SymmetricPositiveDefinite("LEM")

        # Initialize bimap
        bm1 = th.randn((1, self.inputdim, self.dim1), dtype=th.float64)
        bm1 = th.svd(bm1)[0]
        self.register_parameter("bimap1", geoopt.ManifoldParameter(bm1, self.manifold_Stiefel))

        # Initialize Residual Blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(self.n_blocks):
            proj = (i != self.n_blocks - 1)  # proj=False for the last block
            self.residual_blocks.append(
                ResidualBlock(self.dim1, self.manifold_SPD, self.manifold_Stiefel, proj=proj)
            )
                
        # Classifiers
        self.linear_classiflayer = nn.Linear(self.dim1 * self.dim1, self.classes).double()
        self.softmax = nn.Softmax(dim=-1)       

    def forward(self, x):
        """
        Forward pass for single-modality input.
        """
        x = _atxa(self.bimap1, x)
        for res_block in self.residual_blocks:
            x = res_block(x)     

        if self.embed_only:
            return x.reshape(x.shape[0], -1)
        elif self.classifier == "linear":
            return self.softmax(self.linear_classiflayer(x.reshape(x.shape[0], -1))) # softmask required for the Focal loss
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier}. Choose 'linear'.")


class LogEucRResNet_Combo(nn.Module):
    def __init__(self, inputdim_temp, dim1_temp, inputdim_spec, dim1_spec, classes, n_blocks=1, embed_only=False, classifier="linear"):
        """
        Multi-modality ResNet combining temporal and spectral inputs.
        
        Args:
            inputdim_temp (int): Input dimension for temporal modality.
            dim1_temp (int): Dimension after first bi-map for temporal modality.
            inputdim_spec (int): Input dimension for spectral modality.
            dim1_spec (int): Dimension after first bi-map for spectral modality.
            classes (int): Number of output classes.
            n_blocs (int): Number of resiudal blocks 
            embed_only (bool): If True, returns embeddings only.
            classifier (str): Type of classifier ("linear").
        """
        super().__init__()

        self.embed_only = embed_only
        self.classifier = classifier
        self.n_blocks = n_blocks

        # Temporal and Spectral models
        self.model_temp = LogEucRResNet(inputdim_temp, dim1_temp, classes, n_blocks=self.n_blocks, embed_only=True, classifier=self.classifier)
        self.model_spectral = LogEucRResNet(inputdim_spec, dim1_spec, classes, n_blocks=self.n_blocks, embed_only=True, classifier=self.classifier)

        # Fully connected layer
        input_fc_dim = dim1_spec * dim1_spec + dim1_temp * dim1_temp if classifier == "linear" else classes * 2
        self.fc_layer = nn.Linear(input_fc_dim, classes).double()
        self.softmax = nn.Softmax(dim=-1) # softmask required for the Focal loss

    def forward(self, x1, x2):
        """
        Forward pass combining temporal and spectral modalities.
        """
        out_temp = self.model_temp(x1)
        out_spec = self.model_spectral(x2)

        if self.embed_only:
            return th.cat((out_temp, out_spec), dim=1)
        elif self.classifier == "linear":
            x = th.cat((out_temp, out_spec), dim=1)
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier}. Choose 'linear'.")

        return self.softmax(self.fc_layer(x))
    
