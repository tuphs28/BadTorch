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
        device (torch.device): device of layer. Defaults to cpu.
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
        device (torch.device): layer device. Defaults to cpu.
        bias (bool): whether to include bias or not. Defaults to True.
    """

    def __init__(self, in_dim: int, out_dim: int, device: torch.device = torch.device("cpu"), bias: bool = True) -> None:
        # init weights with Xavier initialisation
        self.weight = torch.randn((in_dim, out_dim)).to(device) / (in_dim ** 0.5)
        self.bias = torch.randn((out_dim)).to(device) / (in_dim ** 0.5) if bias else None

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
        device (tprch.device): device to house cell on. Defaults to cpu.
        bias (bool): whether to include bias in RNN cell. Defaults to True.
    """

    def __init__(self, in_dim: int, hidden_dim: int, device: torch.device = torch.device("cpu"), bias: bool = True) -> None:
        # init weights of RNN cell
        self.weight_xh = torch.randn((in_dim, hidden_dim)).to(device)
        self.weight_hh = torch.randn((hidden_dim, hidden_dim)).to(device)
        self.bias = torch.zeros((hidden_dim)).to(device) if bias else None

    def __call__(self, x_t: torch.Tensor, h_t: torch.Tensor, dropout_masks: list = []) -> torch.Tensor:
        if dropout_masks:
            x_t = x_t * dropout_masks[0]
            h_t = h_t * dropout_masks[1]
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
        hidden_dim (int): dimensionality of hidden state of GRU
        device (tprch.device): device to house cell on. Defaults to cpu.
        bias (bool): whether to include bias in GRU cell. Defaults to True.
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

    def __call__(self, x_t: torch.Tensor, h_t: torch.Tensor, dropout_masks: list = []) -> torch.Tensor:
        if dropout_masks:
            x_t = x_t * dropout_masks[0]
            h_t = h_t * dropout_masks[1]
        i_f = torch.sigmoid(x_t @ self.weight_xf + h_t @ self.weight_hf + (self.bias_f if self.bias_f is not None else 0))
        i_o = torch.sigmoid(x_t @ self.weight_xo + h_t @ self.weight_ho + (self.bias_o if self.bias_o is not None else 0))
        h_prop = torch.tanh(x_t @ self.weight_xh + (h_t * i_f) @ self.weight_hh + (self.bias_h if self.bias_h is not None else 0))
        h_t = i_o * h_t + (1 - i_o) * h_prop
        return  h_t

    def parameters(self) -> list:
        weights = [self.weight_xh, self.weight_hh, self.weight_xo, self.weight_ho, self.weight_xf, self.weight_hf]
        biases = [] if self.bias_h is None else [self.bias_h, self.bias_o, self.bias_f]
        return weights + biases
        

class LSTMCell:
    """
    Cell for an LSTM RNN without a peephole that is repeatedly called on each element of an input sequence by a recurrent model

    Args:
        in_dim (int): input dimension
        hidden_dim (int): dimensionality of hidden and cell state of LSTM
        device (torch.device): device to house cell on. Defaults to cpu.
        bias (bool): whether to include bias in LSTM cell. Defaults to True.
    """

    def __init__(self, in_dim: int, hidden_dim: int, device: torch.device = torch.device("cpu"), bias: bool = True) -> None:
        self.hidden_dim = hidden_dim
        # init forget gate weights
        self.weight_xf = torch.randn(size=(in_dim, hidden_dim)).to(device) / (in_dim ** 0.5)
        self.weight_hf = torch.randn(size=(hidden_dim, hidden_dim)).to(device) / (hidden_dim ** 0.5)
        self.bias_f = torch.zeros(size=(hidden_dim,)).to(device) if bias else None
        # init output gate_weights
        self.weight_xi = torch.randn(size=(in_dim, hidden_dim)).to(device) / (in_dim ** 0.5)
        self.weight_hi = torch.randn(size=(hidden_dim, hidden_dim)).to(device) / (hidden_dim ** 0.5)
        self.bias_i = torch.zeros(size=(hidden_dim,)).to(device) if bias else None
        # init output gate_weights
        self.weight_xo = torch.randn(size=(in_dim, hidden_dim)).to(device) / (in_dim ** 0.5)
        self.weight_ho = torch.randn(size=(hidden_dim, hidden_dim)).to(device) / (hidden_dim ** 0.5)
        self.bias_o = torch.zeros(size=(hidden_dim,)).to(device) if bias else None
        # init cell update weights
        self.weight_xc = torch.randn(size=(in_dim, hidden_dim)).to(device) / (in_dim ** 0.5)
        self.weight_hc = torch.randn(size=(hidden_dim, hidden_dim)).to(device) / (hidden_dim ** 0.5)
        self.bias_c = torch.zeros(size=(hidden_dim,)).to(device) if bias else None

    def __call__(self, x_t: torch.Tensor, hc_t: torch.Tensor, dropout_masks: list = []) -> torch.Tensor:
        h_t, c_t = hc_t[:, :, 0], hc_t[:, :, 1]
        if dropout_masks:
            x_t = x_t * dropout_masks[0]
            h_t = h_t * dropout_masks[1]
            c_t = c_t * dropout_masks[1]
        hc_new = torch.zeros_like(hc_t)
        i_f = torch.sigmoid(x_t @ self.weight_xf + h_t @ self.weight_hf + (self.bias_f if self.bias_f is not None else 0))
        i_i = torch.sigmoid(x_t @ self.weight_xi + h_t @ self.weight_hi + (self.bias_i if self.bias_i is not None else 0))
        i_o = torch.sigmoid(x_t @ self.weight_xo + h_t @ self.weight_ho + (self.bias_o if self.bias_o is not None else 0))
        c_prop = torch.tanh(x_t @ self.weight_xc + h_t @ self.weight_hc + (self.bias_c if self.bias_c is not None else 0))
        c_t = i_f * c_t + i_i * c_prop
        h_t = i_o * torch.tanh(c_t)
        hc_new[:, :, 0], hc_new[:, :, 1] = h_t, c_t
        return hc_new

    def parameters(self) -> list:
        weights = [self.weight_xc, self.weight_hc, self.weight_xi, self.weight_hi, self.weight_xo, self.weight_ho, self.weight_xf, self.weight_hf]
        biases = [] if self.bias_c is None else [self.bias_c, self.bias_i, self.bias_o, self.bias_f]
        return weights + biases


class RNN:
    """
    Recurrent layer that repeatedly calls the user-provided RNN cell on each element of an input sequence
        in_dim (int): input dimension
        hidden_dim (int): dimensionality of hidden state of RNN.
        n_layers (int): number of layers of RNN cells to pass input sequence through.
        device (torch.device, optional): device to house cell on. Defaults to cpu.
        bias (bool, optional): whether to include bias in RNN cell. Defaults to True.
        cell_type (bool, optional): type of RNN cell to use out of: RNN, GRU, LSTM. Defaults to RNN.
        dropout (str, optional): type of dropout to use; if none provided, don't use dropout. Defaults to "". Must choose:
            - standard: standard dropout as in (Srivastava et al., 2014) with different mask applied at each timestep
            - variational: variational dropout as in (Gal & Ghahramani, 2015) with same masks applied at each timestep for a sequence
        dropout_strength (bool, optional): dropout probability. Defaults to 0.25.
    Args
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_layers:int, device: torch.device = torch.device("cpu"), bias: bool = True, cell_type: str = "RNN", dropout: str = "", dropout_strength: float = 0.25) -> None:
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        assert cell_type in ["RNN", "GRU", "LSTM"], "Please enter a valid recurrent cell type from: RNN, GRU, LSTM"
        self.cell_type = cell_type
        if self.cell_type == "RNN":
            self.cells = [RNNCell(in_dim, hidden_dim, device, bias) if i==0 else RNNCell(hidden_dim, hidden_dim, device, bias) for i in range(n_layers)]
        elif self.cell_type == "GRU":
            self.cells = [GRUCell(in_dim, hidden_dim, device, bias) if i==0 else GRUCell(hidden_dim, hidden_dim, device, bias) for i in range(n_layers)]
        elif self.cell_type == "LSTM":
            self.cells = [LSTMCell(in_dim, hidden_dim, device, bias) if i==0 else LSTMCell(hidden_dim, hidden_dim, device, bias) for i in range(n_layers)]

    def __call__(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Repeatedly calls RNN cells on each unit of an input sequence. Code looks strange due to requirement that layer must be able to handle LSTM cell states. If using RNN 
        or GRU, h_t is a 3D tensor of hidden states for each batch element for each layer. If using LSTM, h_t is a 4D tensor of hidden states AND cell states or each batch 
        element and each layer. Tracks if model is in training mode for dropout purposes.
        """

        batch_size, seq_length = x.shape[:2]

        # create tensor of prior h_t if not provided
        if h_prev is None:
            if self.cell_type == "LSTM":
                h_prev = torch.zeros(batch_size, self.hidden_dim, self.n_layers, 2).to(self.device) 
            else:
                h_prev = torch.zeros(batch_size, self.hidden_dim, self.n_layers).to(self.device)
        
        # create tensor to store final h_t from each layer
        if self.cell_type == "LSTM":
            h_final = torch.zeros(batch_size, self.hidden_dim, self.n_layers, 2).to(self.device)
        else:
            h_final = torch.zeros(batch_size, self.hidden_dim, self.n_layers).to(self.device)

        inputs = x
        for n_layer in range(self.n_layers):

            cell = self.cells[n_layer]

            # create tensor to store h_t outputs from current layer, and select first h_t to start sequence with
            if self.cell_type == "LSTM":
                h_outputs = torch.zeros((batch_size, seq_length, self.hidden_dim, 2)).to(self.device)
                h_t = h_prev[:, :, n_layer, :]
            else:
                h_outputs = torch.zeros((batch_size, seq_length, self.hidden_dim)).to(self.device)
                h_t = h_prev[:, :, n_layer]

            # create dropout masks for variational dropout or for eval mode - if dropout not used, just create empty list
            if self.dropout == "":
                dropout_masks = []
            elif not training:
                if self.cell_type == "LSTM":
                    dropout_masks = [1 - self.dropout_strength for _ in range(3)]
                else:
                    dropout_masks = [1 - self.dropout_strength for _ in range(2)]
            elif self.dropout == "variational" and  training:
                if self.cell_type == "LSTM":
                    dropout_masks = [(torch.rand((1,dim))>self.dropout_strength).to(torch.float32).to(self.device) for dim in [self.in_dim, self.hidden_dim, self.hidden_dim]]
                else:
                    dropout_masks = [(torch.rand((1,dim))>self.dropout_strength).to(torch.float32).to(self.device) for dim in [self.in_dim, self.hidden_dim]]

            for seq_idx in range(seq_length):
                # create dropout masks for standard dropout
                if self.dropout == "standard" and training:
                    if self.cell_type == "LSTM":
                        dropout_masks = [(torch.rand((1,dim))>self.dropout_strength).to(torch.float32).to(self.device) for dim in [self.in_dim, self.hidden_dim, self.hidden_dim]]
                    else:
                        dropout_masks = [(torch.rand((1,dim))>self.dropout_strength).to(torch.float32).to(self.device) for dim in [self.in_dim, self.hidden_dim]]

                x_t = inputs[:, seq_idx, :]
                h_t = cell(x_t, h_t, dropout_masks)
                if self.cell_type == "LSTM":
                    h_outputs[:, seq_idx, :, :] = h_t # if LSTM, need to select hidden state separate from cell state
                else:
                    h_outputs[:, seq_idx, :] = h_t

            # use hidden states from prior layer as input to following layer
            if self.cell_type == "LSTM":
                inputs = h_outputs[:, :, :, 0] 
            else:
                inputs = h_outputs

            # store hidden states (and cell states if using LSTM) at end of sequence for current layer
            if self.cell_type == "LSTM":
                h_final[:, :, n_layer, :] = h_t
            else:
                h_final[:, :, n_layer] = h_t

        return h_outputs, h_final

    def parameters(self) -> list:
        parameter_list = []
        for cell in self.cells:
            parameter_list += cell.parameters()
        return parameter_list
    

