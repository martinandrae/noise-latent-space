import torch
import numpy as np
from tqdm import tqdm

from sampler import Sampler
from dataset import SQGDataset

fname = 'SQG_0'
bs = 10 # Batch size for inverting the timeseries
invert_steps=500 # This is what was used to generate the current timeseries.
invert_eps=lambda t: 0 * t # No noise when inverting

model_path = "best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_std = 2660
dataset = SQGDataset(f"data/{fname}", mean=0, std=data_std)
loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False)

sampler = Sampler(device, members=1, eps=None, steps=None, invert_eps=invert_eps, invert_steps=invert_steps, model_path=model_path, debug=False)

inverted_dataset = torch.zeros_like(dataset.data)

samples = 0
for i, truth in enumerate(tqdm(loader)):
    truth = truth.to(device)

    inverted_dataset[samples:samples+truth.shape[0]] = sampler.invert(truth).cpu()
    samples += truth.shape[0]

np.save(f"data/inverted_{fname}.npy", inverted_dataset.numpy())