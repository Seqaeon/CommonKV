import torch
import torch.nn as nn
import einops

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
        self.enc1 = nn.Linear(self.feat_dim, self.embed_dim)
        self.act1 = nn.GELU()
        self.out = nn.Linear(self.embed_dim, self.embed_dim)
        self.eps = 1e-6
        self.learnable_scale = torch.nn.Parameter(torch.ones(self.feat_dim))
        self.learnable_scale.requires_grad = False  # freeze learnable scale
        self.CONST = torch.arange(8, dtype=torch.uint8)
        self.CONST1 = (2 ** self.CONST)
        self.CONST2 = (1 << self.CONST)
        self.codebook = None
    

    def move_to(self, device):
        self = self.to(device, non_blocking=True)
        self.embedding = self.embedding.to(device, non_blocking=True)
        self.learnable_scale = torch.nn.Parameter(self.learnable_scale.to(device, non_blocking=True), requires_grad=False)
        if hasattr(self, "codebook"):
            self.codebook = self.codebook.to(device, non_blocking=True)

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

    def cache_postscale(self, x, prescale):
        x = torch.matmul(x, self.embedding)
        postscale = torch.norm(x, dim=-1, keepdim=True)
        return postscale


    def decode2(self, x, prescale):
        x = torch.matmul(x, self.embedding)
        x = x * prescale
        x = x * (self.learnable_scale + self.eps)
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
        x = x.transpose(1, 2)
        shape = x.shape
        dtype = x.dtype
        x = x.flatten().to(torch.uint8)
        x, padding = self.compress_binary_tensor(x)
        assert padding == 0
        x = x.view(shape[0], shape[1], shape[2] // 8).transpose(1, 2)
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
        code = code.transpose(1, 2).flatten()
        x = self.decompress_binary_tensor(code, data["padding"])
        x = x.view(data["shape"]).to(data["dtype"])
        x = x.transpose(1, 2)
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
    def __init__(self, feat_dim, layer_idx, quant_bits=1):
        super().__init__()
        self.feat_dim = feat_dim
        if quant_bits == 1:
            DIR = "Llama-3.1-8B-Instruct-CommVQ-1bit-codebook"
        elif quant_bits == 2:
            DIR = "Llama-3.1-8B-Instruct-CommVQ-2bit-codebook"
        else:
            raise ValueError(f"Unsupported quant_bits: {quant_bits}")
        self.codebook = []
        self.centers = []
        for i in range(8):
            center_data = torch.load(f"{DIR}/{str(layer_idx).zfill(3)}_{i}.pt", map_location="cpu")["steps"]
            clustering_centers = torch.cat([x["clustering_centers"].unsqueeze(0) for x in center_data]).unsqueeze(1)
            self.codebook.append(clustering_centers)
            theta = torch.cat([x["theta"].transpose(0, 1).unsqueeze(0) for x in center_data])
            theta = theta.view(theta.shape[0], theta.shape[1], -1, 2)  # [16, 64, 64, 2] 16residual, 64pairs, 64C (#centers), xy 
            centers = torch.zeros(theta.shape[0], theta.shape[1], theta.shape[2], 2, 2)
            # 
            # [x , y]
            # [-y, x]
            # theta[0]: x, theta[1]: y
            centers[..., 0, 0] =  theta[..., 0]
            centers[..., 1, 1] =  theta[..., 0]
            centers[..., 0, 1] =  theta[..., 1]
            centers[..., 1, 0] = -theta[..., 1]
            self.centers.append(centers.unsqueeze(0))
        self.codebook = torch.cat(self.codebook, dim=1)
        self.centers_legacy = torch.cat(self.centers, dim=0)
        self.centers = self.centers_legacy.repeat_interleave(4, dim=0).unsqueeze(0).unsqueeze(3).transpose(-2, -1).squeeze(0).squeeze(2).permute(0, 2, 1, 3, 4, 5).contiguous()
        del self.centers_legacy
        self.layer_idx = layer_idx


    def move_to(self, device):
        self.codebook = self.codebook.to(device, non_blocking=True)
        # self.centers_legacy = self.centers_legacy.to(device, non_blocking=True)
        self.centers = self.centers.to(device, non_blocking=True)

    def encode(self, x):
        x = x.view(x.shape[0], x.shape[1], 8, 2, 64).transpose(3, 4).flatten(2)
        # self.store_x = x
        prescale = torch.norm(x, dim=-1, keepdim=True)
        x = x / prescale
        tensor = x
        tensor = einops.rearrange(tensor, 'b s (h c) -> (b s) h c', h=self.codebook.shape[1])
        # Initialize the code tensor
        code_tensor = torch.zeros(
            self.codebook.shape[0],  # 8
            self.codebook.shape[1],  # 16
            tensor.shape[0],         # SEQ_LEN
            dtype=torch.int16,
            device=tensor.device
        )
        tensor = tensor.permute(1, 0, 2)
        for r in range(self.codebook.shape[0]):
            codebook_r = self.codebook[r]
            distances = torch.cdist(tensor, codebook_r)
            hard_assignments = torch.argmin(distances, dim=-1)
            quantized_tensor = torch.gather(codebook_r, dim=1, index=hard_assignments.unsqueeze(-1).expand(hard_assignments.shape[0],hard_assignments.shape[1],codebook_r.shape[-1]))
            # Update the tensor
            tensor -= quantized_tensor
            # Store the hard assignments in the corresponding slice of `code_tensor`
            code_tensor[r] = hard_assignments.to(torch.int16)  # Shape: (16, chunk_size)
        return code_tensor, prescale

    def decode(self, x, prescale):
        x = x.long()
        x_expanded = x.unsqueeze(-1)
        x = torch.gather(self.codebook, 2, x_expanded.expand(-1, -1, -1, self.codebook.size(-1)))
        x = x.permute(0, 2, 1, 3).flatten(-2).sum(0)
        x = einops.rearrange(x, '(b s) c -> b s c', b=prescale.size(0))
        x = x * prescale
        return x
    
    def decode2(self, x, prescale):
        x = x.long().permute(1, 0, 2)
        C1_x = x // 64
        C2_x = x % 64
        centers = self.centers_legacy.unsqueeze(2).expand(-1, -1, C1_x.shape[2], -1, -1, -1, -1)
        C1_x = C1_x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 64, -1, 2, 2)
        C2_x = C2_x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 64, -1, 2, 2)
        a = torch.gather(centers, 4, C1_x)
        b = torch.gather(centers, 4, C2_x)
        x = (a[:,:,:,:,:,0] + b[:,:,:,:,:,1]).flatten(-3).sum(1).permute(1, 0, 2).flatten(-2)
        x = einops.rearrange(x, '(b s) c -> b s c', b=prescale.size(0))
        x = x * prescale
        return x

    def pack(self, x, prescale, factor, offload=False):
        B = prescale.shape[0]
        x = x.unflatten(-1, (B, -1))
        # assert offload
        data = {
            "code": x.permute(2, 1, 0, 3).contiguous(),
            "prescale": (prescale.unsqueeze(1).permute(0, 1, 3, 2) / factor).contiguous(),
        }
        if offload:
            data["code"] = data["code"].to("cpu", non_blocking=True)
        return data

    def pack_legacy(self, x, prescale, offload=False):
        # assert offload
        data = {
            "code": x,
            "prescale": prescale,
        }
        if offload:
            data["code"] = data["code"].to("cpu", non_blocking=True)
        return data

    def unpack(self, data, device, offload=False):
        # assert offload
        assert data["code"].dtype == torch.int16
        if offload:
            code = data["code"].to(device)
        else:
            code = data["code"]
        return code, data["prescale"]
