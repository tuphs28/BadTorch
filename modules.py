import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class Embedding:
    """
    Embedding layer that converts vocabulary indices into vectors in embedding space - essentially just a learned lookup table
    
    Args:
        num_emb (int): number of unique items in vocab to embed
        emb_dim (int): dimensionality of embedding space
        device (torch.device): device of layer
    """
    def __init__(self, num_emb: int, emb_dim: int, device: torch.device = torch.device("cpu")) -> None:
        # create table of embeddings where the i-th row corresponds to the embedding of the i-th vocab element
        self.embeddings = torch.randn((num_emb, emb_dim)).to(device)

    def __call__(self, idx: int) -> torch.Tensor:
        x = self.embeddings[idx]
        return x

    def parameters(self) -> list:
        return [self.embeddings]
    

class Linear:
    """
    Basic fully connected layer / MLP layer 

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        device (torch.device): layer device
        bias (bool): whether to include bias or not
    """

    def __init__(self, in_dim: int, out_dim: int, device: torch.device = torch.device("cpu"), bias: bool = True) -> None:
        # init weights
        self.weight = torch.rand((in_dim, out_dim)).to(device) / (in_dim ** 0.5)
        self.bias = torch.zeros((out_dim)).to(device) if bias else None

    def __call__(self, x: torch.tensor) -> torch.tensor:
        h = x @ self.weight
        if self.bias is not None:
            h += self.bias
        return h

    def parameters(self) -> list:
        return [self.weight] + ([] if self.bias is None else [self.bias])
    

class RNNCell:
    """
    Basic RNN cell that is repeatedly called on each element of an input sequence by a recurrent model

    Args:
        in_dim (int): input dimension
        hidden_dim (int): dimensionality of hidden state of RNN
        device (tprch.device): device to house cell on
        bias (bool): whether to include bias in RNN cell
    """

    def __init__(self, in_dim: int, hidden_dim: int, device: torch.device = torch.device("cpu"), bias: bool = True) -> None:
        # init weights of RNN cell
        self.weight_xh = torch.randn((in_dim, hidden_dim)).to(device)
        self.weight_hh = torch.randn((hidden_dim, hidden_dim)).to(device)
        self.bias = torch.zeros((hidden_dim)).to(device) if bias else None

    def __call__(self, x_t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        z_t = x_t @ self.weight_xh + h_t @ self.weight_hh
        if self.bias is not None:
            z_t += self.bias
        h_t = torch.tanh(z_t)
        return h_t

    def parameters(self) -> list:
        return [self.weight_xh, self.weight_hh] + ([] if self.bias is None else [self.bias])


class GRUCell:
    """
    Cell for a Gated Recurrent Unit RNN that is repeatedly called on each element of an input sequence by a recurrent model

    Args:
        in_dim (int): input dimension
        hidden_dim (int): dimensionality of hidden state of RNN
        device (tprch.device): device to house cell on
        bias (bool): whether to include bias in RNN cell
    """

    def __init__(self, in_dim: int, hidden_dim: int, device: torch.device = torch.device("cpu"), bias: bool = True) -> None:
        self.hidden_dim = hidden_dim
        # init forget gate weights
        self.weight_xf = torch.randn(size=(in_dim, hidden_dim)).to(device) / (in_dim ** 0.5)
        self.weight_hf = torch.randn(size=(hidden_dim, hidden_dim)).to(device) / (hidden_dim ** 0.5)
        self.bias_f = torch.zeros(size=(hidden_dim,)).to(device) if bias else None
        # init output gate_weights
        self.weight_xo = torch.randn(size=(in_dim, hidden_dim)).to(device) / (in_dim ** 0.5)
        self.weight_ho = torch.randn(size=(hidden_dim, hidden_dim)).to(device) / (hidden_dim ** 0.5)
        self.bias_o = torch.zeros(size=(hidden_dim,)).to(device) if bias else None
        # init hidden update weights
        self.weight_xh = torch.randn(size=(in_dim, hidden_dim)).to(device) / (in_dim ** 0.5)
        self.weight_hh = torch.randn(size=(hidden_dim, hidden_dim)).to(device) / (hidden_dim ** 0.5)
        self.bias_h = torch.zeros(size=(hidden_dim,)).to(device) if bias else None

    def __call__(self, x_t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        i_f = torch.sigmoid(x_t @ self.weight_xf + h_t @ self.weight_hf + (self.bias_f if self.bias_f is not None else 0))
        i_o = torch.sigmoid(x_t @ self.weight_xo + h_t @ self.weight_ho + (self.bias_o if self.bias_o is not None else 0))
        h_prop = torch.tanh(x_t @ self.weight_xh + (h_t * i_f) @ self.weight_hh + (self.bias_h if self.bias_h is not None else 0))
        h_t = i_o * h_t + (1 - i_o) * h_prop
        return  h_t

    def parameters(self) -> list:
        weights = [self.weight_xh, self.weight_hh, self.weight_xo, self.weight_ho, self.weight_xf, self.weight_hf]
        biases = [] if self.bias_h is None else [self.bias_h, self.bias_o, self.bias_f]
        return weights + biases
        

class RNN:
    """
    RNN layer that repeatedly calls the provided RNN cell on an input sequence to generate an output sequence

    Args:
        in_dim (int): input dimension
        hidden_dim (int): dimensionality of hidden state of RNN
        n_layers (int): number of RNN layers to include
        device (tprch.device): device to house cell on
        bias (bool): whether to include bias in RNN cell
        cell_type (str): string indicating which type of RNN cell to use - current must use either RNN or GRU
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_layers:int, device: torch.device = torch.device("cpu"), bias: bool = True, cell_type: str = "RNN") -> None:
        # set attributes
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # construct RNN layers composed of specified cell type 
        if cell_type == "RNN":
            self.cells = [RNNCell(in_dim, hidden_dim, device, bias) if i==0 else RNNCell(hidden_dim, hidden_dim, device, bias) for i in range(n_layers)]
        elif cell_type == "GRU":
            self.cells = [GRUCell(in_dim, hidden_dim, device, bias) if i==0 else GRUCell(hidden_dim, hidden_dim, device, bias) for i in range(n_layers)]
        else:
            raise ValueError("Invalid RNN cell type provided")

    def __call__(self, x: torch.Tensor, h_prev: Optional[None]) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, seq_length = x.shape[:2]

        # initialise hidden state as zero vector if no hidden state provided
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_dim, self.n_layers).to(self.device)

        inputs = x
        h_final = torch.zeros(batch_size, self.hidden_dim, self.n_layers).to(self.device)
        for n_layer in range(self.n_layers):
            cell = self.cells[n_layer]
            h_outputs = torch.zeros((batch_size, seq_length, self.hidden_dim)).to(self.device)
            h_t = h_prev[:, :, n_layer]
            for seq_idx in range(seq_length):
                x_t = inputs[:, seq_idx, :]
                h_t = cell(x_t, h_t)
                h_outputs[:, seq_idx, :] = h_t
            inputs = h_outputs
            h_final[:, :, n_layer] = h_t

        return h_outputs, h_final

    def parameters(self) -> list:
        parameter_list = []
        for cell in self.cells:
            parameter_list += cell.parameters()
        return parameter_list
    

