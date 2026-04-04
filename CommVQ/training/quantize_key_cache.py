import numpy as np
import torch
import torch.nn.functional as F
import glob
from tqdm import tqdm, trange
import warnings
import math
import random
import json
import sys
import os
from tqdm.contrib.concurrent import process_map
warnings.filterwarnings("ignore")
# Parameters for the EM algorithm
KV_DIM = 1024
N_BITS = 12
N_BITS_HALF = N_BITS // 2
POW_2_N_BITS_HALF = 2**N_BITS_HALF
M_GROUP = 128
RESIDUAL_NUM = 21
RESULT_DIR = f"codebook_{N_BITS}bits_{M_GROUP}group_{RESIDUAL_NUM}residuals"
CLUSTERING_CENTER_NUM = POW_2_N_BITS_HALF * POW_2_N_BITS_HALF
N_STEPS = 100
# Parameters for the temperature scheduler
initial_temp = 1e-3
final_temp = 1e-5
print("CLUSTERING_CENTER_NUM", CLUSTERING_CENTER_NUM)

# T: [2*POW_2_N_BITS_HALF*POW_2_N_BITS_HALF, 2*POW_2_N_BITS_HALF]
# XAB = XA - YB
# YAB = XB + YA
T = torch.zeros(2*POW_2_N_BITS_HALF*POW_2_N_BITS_HALF, 2*POW_2_N_BITS_HALF).cuda()
for A in range(POW_2_N_BITS_HALF):
    for B in range(POW_2_N_BITS_HALF):
        T[2*(A*POW_2_N_BITS_HALF+B), 2*A] = 1
        T[2*(A*POW_2_N_BITS_HALF+B), 2*B+1] = -1
        T[2*(A*POW_2_N_BITS_HALF+B)+1, 2*B] = 1
        T[2*(A*POW_2_N_BITS_HALF+B)+1, 2*A+1] = 1
T = T.float()
print("original T.shape", T.shape)

# T_full = torch.zeros(T.shape[0]*(M_GROUP//2), T.shape[1]*(M_GROUP//2)).cuda()
# for i in range(M_GROUP//2):
#     T_full[i*T.shape[0]:(i+1)*T.shape[0], i*T.shape[1]:(i+1)*T.shape[1]] = T.clone()
# T = T_full.clone()

_temp = T.T @ T
assert torch.sum(_temp - torch.diag(torch.diagonal(_temp))) == 0
print("all good")


class TemperatureScheduler:
    def __init__(self, initial_temp, final_temp, n_steps, strategy='linear'):
        """
        Initialize the temperature scheduler.

        Args:
            initial_temp (float): The starting temperature.
            final_temp (float): The ending temperature.
            n_steps (int): Total number of steps.
            strategy (str): The strategy for updating the temperature.
                            Options: 'linear', 'exponential', 'logarithmic'.
        """
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.n_steps = n_steps
        self.strategy = strategy.lower()

    def get_temperature(self, step):
        """
        Compute the temperature at a given step.

        Args:
            step (int): The current step (0-indexed).

        Returns:
            float: The temperature at the current step.
        """
        if step < 0 or step >= self.n_steps:
            raise ValueError("Step must be within the range [0, n_steps-1].")

        progress = step / (self.n_steps - 1)  # Normalize step to [0, 1]

        if self.strategy == 'linear':
            # Linear decay
            return self.initial_temp + progress * (self.final_temp - self.initial_temp)
        elif self.strategy == 'exponential':
            # Exponential decay
            return self.initial_temp * (self.final_temp / self.initial_temp) ** progress
        elif self.strategy == 'logarithmic':
            # Logarithmic decay (avoid division by zero)
            if step == 0:
                return self.initial_temp
            return self.initial_temp - (self.initial_temp - self.final_temp) * math.log(step + 1) / math.log(self.n_steps)
        else:
            raise ValueError("Unsupported strategy. Choose from 'linear', 'exponential', 'logarithmic'.")



def E_step(tensors, clustering_centers, tensors_norm, temperature=1.0):
    dist = torch.cdist(tensors, clustering_centers)
    soft_assignments = F.softmax(-dist / temperature, dim=1)
    hard_assignments = torch.argmin(dist, dim=1)
    hard_mse_loss = torch.mean((tensors * tensors_norm - clustering_centers[hard_assignments] * tensors_norm) ** 2)
    soft_mse_loss = None
    return soft_assignments, soft_mse_loss, hard_assignments, hard_mse_loss


def M_step(tensors, clustering_centers, soft_assignments):
    assert len(clustering_centers) == CLUSTERING_CENTER_NUM
    # m: new mean of each cluster
    # N_tensors: number of tensors in each cluster
    N_tensors = torch.sum(soft_assignments, dim=0).unsqueeze(-1).repeat(1, 2) + 1e-8
    weighted_sums = torch.matmul(soft_assignments.T, tensors)
    m = weighted_sums / N_tensors[:, :1]
    N_tensors = N_tensors.flatten().unsqueeze(0)
    mat = (T.T * N_tensors) @ T
    A_mat = (torch.linalg.inv(mat) @ T.T) * N_tensors
    B_mat = m.view(m.shape[0], M_GROUP // 2, 2).permute(0, 2, 1).flatten(0, 1)
    theta = A_mat @ B_mat
    clustering_centers = (T @ theta).view(CLUSTERING_CENTER_NUM, 2, M_GROUP // 2).permute(0, 2, 1).flatten(1)
    return clustering_centers, theta


def run_em_algorithm(tensors, tensors_norm, T_scale):
    scheduler = TemperatureScheduler(initial_temp * T_scale, final_temp * T_scale, N_STEPS, strategy='exponential')
    center_num = CLUSTERING_CENTER_NUM
    clustering_centers = tensors[torch.randint(0, N, (center_num,))].squeeze()
    soft_assignments, _, _, _ = E_step(tensors, clustering_centers, tensors_norm=tensors_norm)
    clustering_centers, theta = M_step(tensors, clustering_centers, soft_assignments)
    best_mse_loss = math.inf
    pbar = range(N_STEPS)
    for step in pbar:
        # E step
        temperature = scheduler.get_temperature(step)
        soft_assignments, _, _, mse_loss = E_step(
            tensors, clustering_centers,
            tensors_norm=tensors_norm,
            temperature=temperature,
        )
        # assert mse_loss < best_mse_loss
        if mse_loss < best_mse_loss:
            best_result = {"mse_loss": mse_loss, "theta": theta, "clustering_centers": clustering_centers}
            best_mse_loss = mse_loss
        elif (mse_loss - best_mse_loss) < 1e-5:
            break
        # pbar.set_description(f"T: {temperature:.8f} | best mse loss: {best_mse_loss:.6f}")
        # M step
        clustering_centers, theta = M_step(tensors, clustering_centers, soft_assignments)
    theta = theta.cpu().numpy().tolist()
    best_mse_loss = best_mse_loss.item()
    clustering_centers = clustering_centers.cpu().numpy().tolist()
    return best_result


def quantize_tensor(tensors, clustering_centers):
    dist = torch.cdist(tensors, clustering_centers)
    hard_assignments = torch.argmin(dist, dim=1)
    quantized_tensor = clustering_centers[hard_assignments]
    return quantized_tensor


def handle_single(args, all_tensors, tensors_norm):
    layer_idx, index, N, repeat = args
    filename = os.path.join(RESULT_DIR, f"{str(layer_idx).zfill(3)}_{index}.pt")
    tensors = all_tensors[..., M_GROUP*index:M_GROUP*(index+1)]
    assert tensors.shape[1] == M_GROUP
    assert tensors.shape[0] == N
    all_results = []
    T_scale = 1.0
    for i in range(repeat):
        result = run_em_algorithm(tensors, tensors_norm, T_scale)
        if i % 4 == 0:
            T_scale /= 10.0
        all_results.append(result)
        clustering_centers = torch.tensor(result["clustering_centers"]).cuda()
        quantized_tensor = quantize_tensor(tensors, clustering_centers)
        residuals = tensors - quantized_tensor
        tensors = residuals
        error = torch.mean((residuals * tensors_norm)**2).item()
        print("Error after {}-th residual: {:.4f}".format(i+1, error))
    data = {
        "final_error": error,
        "steps": all_results,
    }
    print("Error for layer {}-th index {}({}-{}): {:.6f}".format(layer_idx, index, index*M_GROUP, (index+1)*M_GROUP, error))
    torch.save(data, filename)
    return error

if __name__ == "__main__":
    os.makedirs(RESULT_DIR, exist_ok=True)
    layer_idx = sys.argv[1] if len(sys.argv) > 1 else 0
    all_tensors = []
    ALL_FILES = glob.glob(f"data/key/{str(layer_idx).zfill(3)}_*.pt")
    ALL_FILES.sort()
    # all_files = ALL_FILES[:4000]
    all_files = ALL_FILES
    for file in all_files:
        tensor = torch.load(file, map_location=torch.device('cpu'))
        all_tensors.append(tensor)
    all_tensors = torch.cat(all_tensors, dim=1).squeeze().float()
    jobs = []
    # for index in [5]:
    for index in range(0, KV_DIM//M_GROUP):
        N = min(256 * CLUSTERING_CENTER_NUM, all_tensors.shape[0])
        repeat=RESIDUAL_NUM
        jobs.append((layer_idx, index, N, repeat))
    all_tensors = all_tensors[:N].cuda()
    all_tensors = all_tensors.view(N, 8, 2, 64).transpose(2, 3).flatten(1)
    tensors_norm = all_tensors.norm(dim=1, keepdim=True)
    all_tensors = all_tensors / tensors_norm
    errors = []
    for job in tqdm(jobs):
        error = handle_single(job, all_tensors, tensors_norm)
        errors.append(error)
    print("Mean error:", np.mean(errors))
