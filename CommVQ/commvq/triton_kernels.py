import torch
import triton
import triton.language as tl
import time

@triton.autotune(
  configs=[triton.Config(kwargs={'BLOCK_SIZE': bs}) for bs in [4, 8, 16, 32]],
  key=["B"],
)
@triton.jit
def calculate_QC_T_kernel(
    query_states_ptr,
    centers_ptr,
    output_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    DIM: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    hx = tl.program_id(0)
    batch_i = hx // H
    hx = hx % H
    dx = tl.program_id(1)
    bx = tl.program_id(2)
    query_start = (hx * DIM * B + dx * B + batch_i) * 2
    q1 = tl.load(query_states_ptr + query_start)
    q2 = tl.load(query_states_ptr + query_start + 1)
    rc = tl.arange(0, BLOCK_SIZE) + bx * BLOCK_SIZE
    mask = rc < (R * C)
    centers_start = (hx * DIM * R * C + dx * R * C + rc) * 4
    c1 = tl.load(centers_ptr + centers_start, mask=mask)
    c2 = tl.load(centers_ptr + centers_start + 1, mask=mask)
    c3 = tl.load(centers_ptr + centers_start + 2, mask=mask)
    c4 = tl.load(centers_ptr + centers_start + 3, mask=mask)
    output_start = (hx * DIM * B * R * C + dx * B * R * C + batch_i * R * C + rc) * 2
    tl.store(output_ptr + output_start, q1 * c1 + q2 * c3, mask=mask)
    tl.store(output_ptr + output_start + 1, q1 * c2 + q2 * c4, mask=mask)


def calculate_QC_T_triton(query_states, centers):
    query_states = query_states.squeeze(2).permute(1, 2, 0, 3).contiguous()
    # query_states: [32, 64, 80, 2]
    # centers: [32, 64, 16, 64, 2, 2]
    H, DIM, R, C, _, _ = centers.shape
    B = query_states.shape[2]
    output = torch.zeros(H, DIM, B, R, C, 1, 2, device=query_states.device, dtype=query_states.dtype)
    grid = lambda META: (B * H, DIM, triton.cdiv(R * C, META["BLOCK_SIZE"]))
    # BLOCK_SIZE = 16
    # grid = (H * B, DIM, triton.cdiv(R * C, BLOCK_SIZE))
    calculate_QC_T_kernel[grid](
        query_states_ptr=query_states,
        centers_ptr=centers,
        output_ptr=output,
        B=B, H=H, DIM=DIM, R=R, C=C,
        # BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


@triton.autotune(
  configs=[triton.Config(kwargs={'BLOCK_SIZE': bs}) for bs in [4, 8, 16, 32]],
  key=['N']
)
@triton.jit
def gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum_kernel(
    key_code_ptr,
    QC_T_ptr,
    sin_ptr,
    cos_ptr,
    prescale_ptr,
    output_ptr,
    N: int,
    H_RATIO: tl.constexpr, KV_H: tl.constexpr, DIM: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    hx = tl.program_id(0)
    h_i = tl.program_id(1)
    bx = tl.program_id(2)
    token_offset = tl.arange(0, BLOCK_SIZE) + bx * BLOCK_SIZE
    mask = token_offset < N
    dx = tl.arange(0, DIM)
    # sin_start: [BLOCK_SIZE, DIM]
    sin_start = token_offset[:, None] * DIM + dx[None, :]
    cos_start = token_offset[:, None] * DIM + dx[None, :]
    prescale = tl.load(prescale_ptr + token_offset, mask=mask)
    codebook_start = (hx * DIM + dx) * (H_RATIO * R * C * 2)
    acc_1X = tl.zeros([BLOCK_SIZE, DIM], dtype=tl.bfloat16)
    acc_1Y = tl.zeros([BLOCK_SIZE, DIM], dtype=tl.bfloat16)
    acc_2X = tl.zeros([BLOCK_SIZE, DIM], dtype=tl.bfloat16)
    acc_2Y = tl.zeros([BLOCK_SIZE, DIM], dtype=tl.bfloat16)
    for r_i in tl.static_range(0, R):
        c_start = hx * R * N + r_i * N + token_offset
        c = tl.load(key_code_ptr + c_start)  # FIXME: do I need mask?
        c1x = c // 64
        c2x = c % 64
        c1_offset = codebook_start[None, :] + h_i * R * C * 2 + r_i * C * 2 + c1x[:, None] * 2
        c2_offset = codebook_start[None, :] + h_i * R * C * 2 + r_i * C * 2 + c2x[:, None] * 2
        c1x = tl.load(QC_T_ptr + c1_offset)
        c1y = tl.load(QC_T_ptr + c1_offset + 1)
        c2x = tl.load(QC_T_ptr + c2_offset)
        c2y = tl.load(QC_T_ptr + c2_offset + 1)
        acc_1X += c1x
        acc_1Y += c1y
        acc_2X += c2x
        acc_2Y += c2y
    sin = tl.load(sin_ptr + sin_start)
    cos = tl.load(cos_ptr + cos_start)
    acc = tl.sum(acc_1X * cos + acc_1Y * sin + acc_2Y * cos - acc_2X * sin, axis=1)
    output_offset = (hx * H_RATIO + h_i) * N + token_offset
    acc = acc * prescale
    tl.store(output_ptr + output_offset, acc, mask=mask)

def gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum_triton(key_code, QC_T, qsin, qcos, prescale, factor, H_RATIO=4):
    B = prescale.shape[0]
    QC_T = QC_T.unflatten(0, (-1, H_RATIO)).permute(3, 0, 2, 1, 4, 5, 6, 7).flatten(0, 1).contiguous()
    key_code =key_code.flatten(0, 1).contiguous()
    # key_code = key_code.transpose(0, 1)
    # QC_T: [B * KV_H, DIM, H_RATIO, R, C, 1, 2]
    # key_code: [B * KV_H, R, N]
    # qsin: [1, 1, N, DIM, 1]
    # qcos: [1, 1, N, DIM, 1]
    # output X: [KV_H * H_RATIO, N, DIM, 2]
    # output Y: [KV_H * H_RATIO, N, DIM, 2]
    # output: [1, KV_H * H_RATIO, N]
    assert key_code.shape[0] == B * 8
    assert QC_T.shape[0] == B * 8 and QC_T.shape[1] == 64 and QC_T.shape[2] == H_RATIO and QC_T.shape[4] == 64 and QC_T.shape[5] == 1 and QC_T.shape[6] == 2
    KV_H, _, N = key_code.shape
    _, DIM, _, R, C, _, _ = QC_T.shape
    # assert N % BLOCK_SIZE == 0
    assert H_RATIO == 4
    output = torch.zeros(KV_H * H_RATIO, N, device=QC_T.device, dtype=QC_T.dtype)
    # BLOCK_SIZE = 32
    # grid = (KV_H, H_RATIO, triton.cdiv(N, BLOCK_SIZE))
    grid = lambda META: (KV_H, H_RATIO, triton.cdiv(N, META["BLOCK_SIZE"]))
    gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum_kernel[grid](
        key_code_ptr=key_code,
        QC_T_ptr=QC_T,
        sin_ptr=qsin,
        cos_ptr=qcos,
        prescale_ptr=prescale,
        output_ptr=output,
        N=N, 
        H_RATIO=H_RATIO, KV_H=KV_H, DIM=DIM, R=R, C=C,
        # BLOCK_SIZE=BLOCK_SIZE,
    )
    output = output.unflatten(0, (B, -1))
    return output

def gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum(key_code, QC_T, qsin, qcos, prescale, factor):
    attn_weights1 = gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum_triton(key_code, QC_T, qsin, qcos, prescale, factor)
    attn_weights1 = attn_weights1.unsqueeze(-2)
    return attn_weights1


configs = []
for bs in [4, 8, 16, 32]:
    for hs in [16, 32, 64]:
        for ds in [16, 32, 64]:
            configs.append(triton.Config(kwargs={
                'BLOCK_SIZE': bs,
                "H_SIZE": hs,
                "D_SIZE": ds,
            }))

@triton.autotune(
  configs=configs,
  key=['N']
)
@triton.jit
def attn_weights_mul_value_kernel(
    attn_weights_ptr,
    value_code_ptr,
    SA_ptr,
    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    N: int,
    BLOCK_SIZE: tl.constexpr, H_SIZE: tl.constexpr, D_SIZE: tl.constexpr,
):
    # attn_weights: [B, H, 1, N * 8]
    # value_code: [B, 1, N, D]
    # SA:[B, H, D]
    hx = tl.program_id(0)
    batch_i = hx >> 1
    hx = hx & 1
    dx = tl.program_id(1)
    bx = tl.program_id(2)
    hx_i = tl.arange(0, H_SIZE) + hx * H_SIZE
    dx_i = tl.arange(0, D_SIZE) + dx * D_SIZE
    i = tl.arange(0, BLOCK_SIZE) + bx * BLOCK_SIZE
    mask = i < N
    # code: [BLOCK_SIZE, D_SIZE]
    code_start = batch_i * N * D + i[:, None] * D + dx_i[None, :]
    code = tl.load(value_code_ptr + code_start, mask=mask[:, None])
    # attn_weight_start: [H_SIZE, BLOCK_SIZE]
    attn_weight_start = batch_i * H * N * 8 + hx_i[:, None] * N * 8 + i[None, :] * 8
    # offset: [8]
    offset = tl.arange(0, 8)
    # attn_weight: [H_SIZE, BLOCK_SIZE, 8]
    attn_weight = tl.load(attn_weights_ptr + attn_weight_start[:, :, None] + offset[None, None, :], mask=mask[None, :, None])
    # bit: [BLOCK_SIZE, 8, D_SIZE]
    bit = (((code[:, None, :] & (1 << offset[None, :, None])) != 0) * 1.0).cast(tl.bfloat16)
    # attn_weight: [H_SIZE, BLOCK_SIZE * 8]
    attn_weight = tl.reshape(attn_weight, (H_SIZE, attn_weight.shape[1] * attn_weight.shape[2]))
    # bit: [BLOCK_SIZE * 8, D_SIZE]
    bit = tl.reshape(bit, (bit.shape[0] * bit.shape[1], D_SIZE))
    acc_fp16 = tl.dot(attn_weight, bit).cast(tl.float16)
    # output_start: [H_SIZE, D_SIZE]
    output_start = batch_i * H * D + hx_i[:, None] * D + dx_i[None, :]
    tl.atomic_add(SA_ptr + output_start, acc_fp16)


def attn_weights_mul_value_triton(attn_weights, value_code, prescale, learnable_scale, eps, codebook):
    attn_weights = (attn_weights[..., :prescale.shape[1]] * prescale.unsqueeze(1).transpose(-2, -1)).contiguous()
    value_code = value_code.unsqueeze(1).contiguous()
    assert value_code.shape[-1] == 1024, "Currently only 1-bit quantization is supported to use Triton kernels."
    # triton code to impl the following operation
    # SA = torch.matmul(attn_weights, value_code)
    B, H, N, D = attn_weights.shape[0], attn_weights.shape[1], value_code.shape[2], value_code.shape[3]
    # assert B == 1
    # BLOCK_SIZE = 128
    # H_SIZE = 32
    # D_SIZE = 32
    # assert (B * H) % H_SIZE == 0
    # assert D % D_SIZE == 0
    # assert H == H_SIZE * 2
    # grid = ((B * H) // H_SIZE, D // D_SIZE, triton.cdiv(N, BLOCK_SIZE))
    grid = lambda META: (triton.cdiv(B * H, META["H_SIZE"]), triton.cdiv(D, META["D_SIZE"]), triton.cdiv(N, META["BLOCK_SIZE"]))
    SA = torch.zeros(B, H, D, device=attn_weights.device, dtype=torch.float16)
    attn_weights_mul_value_kernel[grid](
        attn_weights_ptr=attn_weights,
        value_code_ptr=value_code,
        SA_ptr=SA,
        B=B, H=H, D=D, N=N,
        # BLOCK_SIZE=BLOCK_SIZE,
        # H_SIZE=H_SIZE,
        # D_SIZE=D_SIZE,
    )
    SA = SA.unsqueeze(2).to(torch.bfloat16)
    SA = SA * (learnable_scale + eps)
    attn_output1 = torch.matmul(SA, codebook.unsqueeze(0))
    return attn_output1

