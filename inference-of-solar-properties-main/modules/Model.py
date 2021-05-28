import torch
import torch.nn as nn
import numpy as np
from modules.Siren import SirenNet, MappingNetwork
from modules.Transformer import Encoder

class FiLM(nn.Module):
    def __init__(self, n_input=128, n_output=128):
        """
        Feature-wise Linear Modulation layer
        """
        super(FiLM, self).__init__()
                                
        self.to_gamma = nn.Linear(n_input, n_output)
        self.to_beta = nn.Linear(n_input, n_output)
                
    def forward(self, x):

        return self.to_gamma(x), self.to_beta(x)

    def weights_init(self):
        for module in self.modules():
            kaiming_init(module)

class DecoderSiren(nn.Module):
    def __init__(self, dim_input = 1, dim_condition=155, dim_output=1, dim_hidden = 128, dim_hidden_mapping = 128, siren_num_layers=6):
        """
        Decoder based on the SIREN network
        It contains a simple mapping network that couples the size of the condition vector to the internal size on the SIREN
        """
        super(DecoderSiren, self).__init__()

        # Small network that maps the input conditioning vector to that inside the SIREN
        self.mapping = MappingNetwork(dim_input=dim_condition, dim_hidden=dim_hidden_mapping, dim_out = dim_hidden, depth_hidden = 2)

        # The SIREN itself
        self.siren = SirenNet(dim_in = dim_input, dim_hidden = dim_hidden, dim_out = dim_hidden, num_layers = siren_num_layers)

        # Output linear projection to the desired output size
        self.to_T = nn.Linear(dim_hidden, dim_output)
                        
    def forward(self, x, latent):
        
        # Get the FiLM vectors
        gamma, beta = self.mapping(latent)
        
        # Call the SIREN with the input x and the conditioning computed with the mapping
        # One can potentially use a different mapping for each layer but this requires changing the SIREN
        out = self.siren(x, gamma, beta)

        # Get the output
        out = self.to_T(out)

        # Remove any possible remaining dimension with size 1
        return out.squeeze()

class TransformerEncoder(nn.Module):
    """Container module with a transformer encoder"""

    def __init__(self, hyperparameters):
        super(TransformerEncoder, self).__init__()
        
        self.hyper = hyperparameters

        self.n_input = self.hyper['n_input']
        self.embed_dim = self.hyper['embed_dim']
        self.num_heads = self.hyper['num_heads']
        self.num_layers = self.hyper['num_layers']
        self.ff_dim = self.hyper['ff_dim']
        self.dropout = self.hyper['dropout']
        self.norm_in = self.hyper['norm_in']
        self.latent_dim = self.hyper['latent_dim']
        self.num_siren_layers = self.hyper['num_siren_layers']
        self.weight_init_type = self.hyper['weight_init_type']        

        self.model_type = 'Transformer'        
        self.src_mask = None

        # ---------------------------------
        # Encoder: embedding + position embedding + transformer encoder
        # ---------------------------------

        # Embedding
        self.input_embedding = nn.Linear(self.n_input, self.embed_dim)
        
        # Wavelength embedding
        self.wavelength_embedding = FiLM(n_input = 1, n_output = self.embed_dim)

        # Transformer encoder
        self.transformer_encoder = Encoder(self.num_layers, self.num_heads, self.embed_dim, self.ff_dim, dropout=self.dropout, norm_in=self.norm_in)

        # ---------------------------------
        # Decoder: SIREN conditioned on the encoder latent vector
        # ---------------------------------        
        self.decoder = DecoderSiren(dim_input=1, dim_condition=self.embed_dim, dim_output=6, dim_hidden=128, dim_hidden_mapping=self.latent_dim, siren_num_layers=self.num_siren_layers)

        self.init_weights()

    def forward(self, wavelength, stokes, mask, tau):

        # Compute the embedding of the input using the linear layer
        input_embed = self.input_embedding(stokes)

        # Compute the wavelength embedding using a FiLM layer
        gamma, beta = self.wavelength_embedding(wavelength[:, :, None])

        # Now we FiLM the input embedding
        input_embed = input_embed * gamma + beta

        # Expand the mask
        src_mask = mask.unsqueeze(1).unsqueeze(2)

        # Produce the encoding for each observation using the transformer encoder
        encoding_all_wavel = self.transformer_encoder(input_embed, src_mask)

        # Average them all (or take first?)
        latent = torch.mean(encoding_all_wavel, dim=1)
        
        output = self.decoder(tau[:, :, None], latent)

        return output

    def init_weights(self):
        init_func = nn.init.xavier_normal_ if self.weight_init_type == 'xavier_normal' else nn.init.xavier_uniform_
        for m in [self.transformer_encoder.self_atts, self.transformer_encoder.pos_ffs]:
            for p in m.parameters():
                if p.dim() > 1:
                    init_func(p)
                else:
                    nn.init.constant_(p, 0.)
