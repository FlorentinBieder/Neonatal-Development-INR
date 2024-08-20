import torch
import torch.nn
from nn.activation_functions import Sine, WIRE, Relu, ComplexWIRE
from nn.pos_encoding import PosEncodingNeRFOptimized, PosEncodingGaussian, PosEncodingNone, PosEncodingNeRF
#from nn.models import ResMLP, ResMLPHiddenCoords
import einops


class Network(torch.nn.Module):
    def __init__(
            self,
            latent_size,        # number of dimensions of latent code
            latent_count,       # number of latent codes
            hidden_size,
            input_size,         # number of input dimensions (x,y,z,t)
            num_layers,
            output_size,
            activation,
            mlp_type,
            pos_encoding,
            num_frequencies,
            embedding_max_norm,
            embedding_norm_type,
    ):
        super().__init__()
        self.mlp_type = mlp_type
        activation_module = dict(
            sine=Sine,
            wire=WIRE,
            relu=Relu,
            complexwire=ComplexWIRE
        )[activation]
        self.position_encoding = dict(
            none=PosEncodingNone,
            nerfoptimized=PosEncodingNeRFOptimized,
            gaussian=PosEncodingGaussian,
            nerf=PosEncodingNeRF,
        )[pos_encoding](input_size=input_size, num_frequencies=num_frequencies)

        self.latent_embeddings = torch.nn.Embedding(
            num_embeddings=latent_count,
            embedding_dim=latent_size,
            padding_idx=0,
            max_norm=embedding_max_norm,
            norm_type=embedding_norm_type,
        )
        with torch.no_grad():
            self.latent_embeddings.weight *= 0 #initialize with zero

        self.layers = torch.nn.ModuleList()
        self.layers.append(activation_module(input_size+latent_size, hidden_size))
        self.layers.extend([activation_module(hidden_size, hidden_size) for k in range(num_layers-2)])
        self.layers.append(activation_module(hidden_size, output_size))


    def forward(self, txyz, latent_idx=None, custom_latent=None):
        x = self.position_encoding(txyz)
        if latent_idx is not None:
            h = self.latent_embeddings(latent_idx)
        elif custom_latent is not None:
            h = custom_latent
        else:
            raise ValueError(f'latent_idx and custom_latent cannot simulataneously be None')
        expanded_h = einops.repeat(h, 'b l -> b p l', p=x.shape[1])
        new_x = torch.concat([x, expanded_h], dim=-1)
        x = new_x
        last_k = 0
        for k, layer in enumerate(self.layers):
            x_prev = x
            x = layer(x)
            if self.mlp_type == 'fullresidual':
                if layer.in_size == layer.out_size:
                    x = (x + x_prev)/2
            elif self.mlp_type == 'skip2residual':
                if layer.in_size == layer.out_size and k >= last_k+2:
                    last_k = k
                    x = (x + x_prev)/2
            elif self.mlp_type == 'none':
                pass
            else:
                raise NotImplementedError()
                #print(self.mlp_type)
        return x




        pass
