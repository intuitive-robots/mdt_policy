from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.models as models
import einops

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
MEAN_MIN = -9.0
MEAN_MAX = 9.0



def set_parameter_requires_grad(model, requires_grad):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = requires_grad
            
            
def freeze_params(model):
    set_parameter_requires_grad(model, requires_grad=False)
    

def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


class SpatialSoftmax(nn.Module):
    def __init__(self, num_rows: int, num_cols: int, temperature: Optional[float] = None):
        """
        Computes the spatial softmax of a convolutional feature map.
        Read more here:
        "Learning visual feature spaces for robotic manipulation with
        deep spatial autoencoders." Finn et al., http://arxiv.org/abs/1509.06113.
        :param num_rows:  size related to original image width
        :param num_cols:  size related to original image height
        :param temperature: Softmax temperature (optional). If None, a learnable temperature is created.
        """
        super(SpatialSoftmax, self).__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, num_cols), torch.linspace(-1.0, 1.0, num_rows), indexing="ij"
        )
        x_map = grid_x.reshape(-1)
        y_map = grid_y.reshape(-1)
        self.register_buffer("x_map", x_map)
        self.register_buffer("y_map", y_map)
        if temperature:
            self.register_buffer("temperature", torch.ones(1) * temperature)
        else:
            self.temperature = Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        x = x.contiguous().view(-1, h * w)  # batch, C, W*H
        softmax_attention = F.softmax(x / self.temperature, dim=1)  # batch, C, W*H
        expected_x = torch.sum(self.x_map * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.y_map * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat((expected_x, expected_y), 1)
        self.coords = expected_xy.view(-1, c * 2)
        return self.coords  # batch, C*2



class BesoResNetEncoder(nn.Module):

    def __init__(
        self,
        latent_dim: int = 128,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        use_mlp: bool = True,
        device: str = 'cuda:0'
    ):
        super(BesoResNetEncoder, self).__init__()
        self.latent_dim = latent_dim
        backbone = models.resnet18(pretrained=pretrained)
        n_inputs = backbone.fc.in_features
        modules = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        if freeze_backbone:
            freeze_params(self.backbone)
        
        # subsitute norm for ema diffusion stuff
        replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
        self.use_mlp = use_mlp
        if self.use_mlp:
            self.fc_layers = nn.Sequential(nn.Linear(n_inputs, latent_dim))

    def conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x):
        batch_size = len(x)
        t_steps = 1
        time_series = False
        
        if len(x.shape) == 5:
            t_steps = x.shape[1]
            x = einops.rearrange(x, 'b t n x_dim y_dim -> (b t) n x_dim y_dim')
            # print(f'After rearrange x shape: {x.shape}')
            time_series = True
        
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        x = self.conv_forward(x)
        if self.use_mlp:
            x = self.fc_layers(x)
        
        if time_series:
            x = einops.rearrange(x, '(b t) d -> b t d', b=batch_size, t=t_steps, d=self.latent_dim)        
        return x

