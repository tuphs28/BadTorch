import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import Embedding, RNN, Linear

class RecurrentLanguageModel:
    """
    Class for constructing recurrent character-level language models

    Args:
        vocab_size (int): number of unique chars in our data.
        emb_dim (int): dimensionality of space in which to embed chars before passing in to model.
        hidden_dim (int): dimensionality of hidden state (and cell state for LSTM).
        n_layers (int): number of recurrent layers to include.
        device (torch.device, optional): device to house model on. Defaults to CPU.
        bias (bool, optional): whether to include biases in layers. Defaults to True.
        cell_type (str, optional): type of RNN cell to use out of:
            - RNN: standard RNN cell consisting of basic hidden-to-hidden updates.
            - GRU: Gated recurrent units with output and forget gates.
            - LSTM: Long short term memory units with input, output and forget gates (non-peephole config)
        dropout (str, optional): type of dropout to use; if none provided, don't use dropout. Defaults to "". Must choose:
            - standard: standard dropout as in (Srivastava et al., 2014) with different mask applied at each timestep
            - variational: variational dropout as in (Gal & Ghahramani, 2015) with same masks applied at each timestep for a sequence
        dropout_strength (bool, optional): dropout probability. Defaults to 0.25.
    """

    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, n_layers: int, device: torch.device = torch.device("cpu"), bias: bool = True, cell_type: str = "RNN",  dropout: str = "", dropout_strength: bool = 0.25):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.cell_type = cell_type
        self.dropout = dropout
        self.dropout_strength = dropout_strength

        self.embedding = Embedding(vocab_size, emb_dim, device)
        self.rnn = RNN(emb_dim, hidden_dim, n_layers, device, bias, cell_type, dropout, dropout_strength)
        self.linear = Linear(hidden_dim, vocab_size, device, bias)

        for parameter in self.parameters():
            parameter.requires_grad = True

        self.training = True # tracks whether model is in train or eval mode for dropout

    def __call__(self, x, h = None):

        embs = self.embedding(x)
        hs, h_final = self.rnn(embs, h, self.training)
        if self.cell_type == "LSTM": # if using LSTM cells hs contains both hidden and cell states, so need to select hidden state only to pass into next layer
            hs = hs[:,:,:,0]

        if self.training and self.dropout == "standard": # if using standard dropout, need to generate a new mask for each timestep and so have to sequentially pass timestep logits through linear layer
            logits = torch.zeros(size=(hs.shape[0]*hs.shape[1], self.vocab_size), device=self.device)
            for seq_idx in range(hs.shape[1]):
                seq_hs = hs[:, [seq_idx], :].view(-1, self.hidden_dim).to(self.device)
                seq_logits = self.linear(seq_hs, self.training)
                logits[torch.arange(seq_idx, hs.shape[0]*hs.shape[1], hs.shape[1]), :] = seq_logits
        else:
            hs = hs.contiguous().view(-1, self.hidden_dim).to(self.device)
            logits = self.linear(hs, self.training)

        return logits, h_final

    def parameters(self) -> None:
        parameters = []
        parameters += self.embedding.parameters()
        parameters += self.rnn.parameters()
        parameters += self.linear.parameters()
        return parameters
    
    def sample(self, idx_to_char_dict: dict, char_to_idx_dict: dict, prompt: str = "", sample_length: int = 250, decoding_method: str = "sample", beam_width: int = 3, k: int = 10) -> str:
        """Sampled from recurrent language model using current weights with various decoding methods.

        Args:
            idx_to_char_dict (dict): dictionary mapping from index to char
            char_to_idx_dict (_type_): dictionary mapping from char to index
            prompt (str, optional): model prompt to begin decoding with. Defaults to "".
            sample_length (int, optional): _number of chars to sample. Defaults to 250.
            decoding_method (str, optional): decoding method to use out of ["greedy", "sample", "top-k", "beam]. Defaults to "sample".
            beam_width (int, optional): beam width paramter is decoding_method is set to "beam". Defaults to 3.
            k (int, optional): size of pruned output distribution if decoding_method is set to "top-k". Defaults to 10.

        Returns:
            str: string of output text sampled from model using the provided decoding method
        """

        current_mode = self.training
        self.training = False

        history = []
        hidden = None
        if prompt:
            for char in prompt:
                current_idx = char_to_idx_dict[char]
                history.append(current_idx)
                current_idx = torch.Tensor([[current_idx]]).int().to(self.device)
                out, hidden = self(current_idx, hidden)
        else:
            current_idx = torch.randint(high=self.vocab_size, size=(1,1)).to(self.device)
            history.append(current_idx.item())

        if decoding_method == "sample": # decode by merely sampling from output distribution at each step

            for i in range(sample_length):
                out, hidden = self(current_idx, hidden)
                with torch.no_grad():
                    prob = nn.functional.softmax(out[-1], dim=0).data
                current_idx = torch.multinomial(prob, 1).view(1,1).to(self.device)
                history.append(current_idx.item())
            sampled_text = []
            for idx in history:
                sampled_text.append(idx_to_char_dict[idx])
            sampled_text = "".join(sampled_text)

        elif decoding_method == "greedy": # decode by taking the most porbably output at each step

            for _ in range(sample_length):
                out, hidden = self(current_idx, hidden)
                with torch.no_grad():
                    prob = nn.functional.softmax(out[-1], dim=0).data
                current_idx = prob.view(-1).argmax().view(1,1)
                history.append(current_idx.item())
            sampled_text = []
            for idx in history:
                sampled_text.append(idx_to_char_dict[idx])
            sampled_text = "".join(sampled_text)

        elif decoding_method == "top-k": # decode by redistributing probability mass over the K most probable outputs and sampling

            for _ in range(sample_length):
                out, hidden = self(current_idx, hidden)
                with torch.no_grad():
                    prob = nn.functional.softmax(out[-1], dim=0).data
                top_prob = torch.topk(prob, k)
                current_idx = top_prob.indices[torch.multinomial(top_prob.values, 1)].view(1,1)
                history.append(current_idx.item())
            sampled_text = []
            for idx in history:
                sampled_text.append(idx_to_char_dict[idx])
            sampled_text = "".join(sampled_text)

        elif decoding_method == "beam": # decode with beam search

            beams = []
            current_str = "".join([idx_to_char_dict[i] for i in history])
            beams = [
                [current_str, 1, hidden]
            ]
            k = beam_width

            for _ in range(sample_length):
                new_beams = []
                for beam in beams:
                    current_str, prob, hidden = beam
                    current_idx = torch.Tensor([char_to_idx_dict[current_str[-1]]]).int().view(1,1).to(self.device)
                    logits, hidden = self(current_idx, hidden)
                    probs = F.softmax(logits.view(-1), dim=0)
                    for prop_idx, prop_prob in zip(torch.topk(probs, k).indices.view(-1), torch.topk(probs, k).values.view(-1)):
                        new_prob = prob * prop_prob.item()
                        new_str = current_str + idx_to_char_dict[prop_idx.item()]
                        if new_str[-3:].count(" ") > 2: # penalise excessive outputing of spaces (needed due to how text files are formatted on laptop)
                            new_prob *= 0.1**(new_str[-3:].count(" ")-1)
                        if len(new_beams) < k:
                            new_str = current_str + idx_to_char_dict[prop_idx.item()]
                            new_beams.append([new_str, new_prob, hidden])
                        else:
                            sorted_probs = [new_beam[1] for new_beam in new_beams]
                            sorted_probs.sort(reverse=True)
                        if new_prob > sorted_probs[k-1]:
                            new_str = current_str + idx_to_char_dict[prop_idx.item()]
                            for i in range(len(new_beams)):
                                if sorted_probs[k-1] == new_beams[i][1]:
                                    new_beams = new_beams[:i] + [[new_str, new_prob, hidden]] + new_beams[i+1:]
                beams = new_beams
            max_prob = -1
            for beam in beams:
                if beam[1] > max_prob:
                    max_prob = beam[1]
                    sampled_text = beam[0]

        self.training = current_mode

        return sampled_text