import torch
import torch.nn as nn
import einops
import glob

class ValueCompressor(nn.Module):
    def __init__(self, quant_bits, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim
        self.quant_bits = quant_bits
        self.code_bit = 2
        self.code_range = 2 ** self.code_bit
        self.n_e = int(self.feat_dim * self.quant_bits) // self.code_bit
        self.embed_dim = self.n_e * self.code_range
        self.embedding = torch.nn.Parameter(torch.randn(self.n_e * self.code_bit, self.feat_dim))
        self.norm1 = nn.LayerNorm(self.feat_dim)
        self.intermediate_dim = min(4096, self.embed_dim) if self.embed_dim != 6144 else self.embed_dim
        self.enc1 = nn.Linear(self.feat_dim, self.intermediate_dim)
        self.act1 = nn.GELU()
        self.out = nn.Linear(self.intermediate_dim, self.embed_dim)
        self.eps = 1e-6
        self.learnable_scale = torch.nn.Parameter(torch.ones(self.feat_dim))
        self.learnable_scale.requires_grad = False  # freeze learnable scale
        self.CONST = torch.arange(8, dtype=torch.uint8)
        self.CONST1 = (2 ** self.CONST)
        self.CONST2 = (1 << self.CONST)

    def encode(self, x):
        original_bits = x.shape[-1] * 16
        compressed_bit = self.embedding.shape[0]
        compression_ratio = original_bits // compressed_bit
        assert int(16 / self.quant_bits) == compression_ratio
        x = x / (self.learnable_scale + self.eps)
        prescale = torch.norm(x, dim=-1, keepdim=True)
        x = x / (prescale + self.eps)
        x = self.norm1(x)
        x = self.enc1(x)
        x = self.act1(x)
        x = self.out(x)
        x = self.quantize(x)
        return x, prescale

    def quantize(self, x):
        B, S, E = x.shape
        x = x.reshape(B, S, self.n_e, self.code_range)
        x = torch.nn.functional.gumbel_softmax(x, tau=1.0, hard=True)
        x[..., :2] += x[..., 2:3]
        x = x[..., :2].flatten(2)
        return x

    def decode(self, x, prescale):
        x = torch.matmul(x, self.embedding)
        postscale = torch.norm(x, dim=-1, keepdim=True)
        x = x / (postscale + self.eps)
        x = x * (prescale + self.eps)
        x = x * (self.learnable_scale + self.eps)
        return x

    def pack(self, x, prescale, offload=False):
        # assert offload
        shape = x.shape
        dtype = x.dtype
        x = x.flatten().to(torch.uint8)
        x, padding = self.compress_binary_tensor(x)
        assert padding == 0
        data = {
            "code": x,
            "padding": padding,
            "shape": list(shape),
            "prescale": prescale,
            "dtype": dtype,
        }
        if offload:
            data["code"] = data["code"].to("cpu", non_blocking=True)
        return data

    def unpack(self, data, device, offload=False):
        # assert offload
        assert data["code"].dtype == torch.uint8
        if offload:
            code = data["code"].to(device)
        else:
            code = data["code"]
        x = self.decompress_binary_tensor(code, data["padding"])
        x = x.view(data["shape"]).to(data["dtype"])
        return x, data["prescale"]

    def compress_binary_tensor(self, binary_tensor):
        assert binary_tensor.dtype == torch.uint8
        assert torch.all((binary_tensor == 0) | (binary_tensor == 1)), "Tensor values must be 0 or 1"

        padding = (8 - binary_tensor.numel() % 8) % 8
        if padding > 0:
            binary_tensor = torch.cat([binary_tensor, torch.zeros(padding, dtype=torch.uint8)])

        binary_tensor = binary_tensor.view(-1, 8)
        packed_tensor = (binary_tensor * self.CONST1).sum(dim=1)
        packed_tensor = packed_tensor.to(torch.uint8)
        return packed_tensor, padding


    def decompress_binary_tensor(self, packed_tensor, padding):
        binary_tensor = ((packed_tensor.unsqueeze(1) & self.CONST2) > 0).to(torch.uint8)
        binary_tensor = binary_tensor.view(-1)[:binary_tensor.numel() - padding]
        return binary_tensor


class KeyCompressor(nn.Module):
    def __init__(self, feat_dim, layer_idx, quant_bits=1, config=None):
        super().__init__()
        return
