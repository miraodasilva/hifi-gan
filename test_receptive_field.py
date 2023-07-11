from models import Generator
from env import AttrDict
import json
import torch

cfg_str = "./config_v1_honglie.json"

with open(cfg_str) as f:
    data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

device = torch.device("cuda:0")
generator = Generator(h, causal=True, bias=False).to(device)

## We test if the network is causal by feeding in half a second of zeros and half a second of 1s
melspec = torch.cat([torch.ones(1, 80, 1), torch.zeros(1, 80, 99)], dim=-1).to(device)

wav = generator(melspec)
first_zero_index = torch.count_nonzero(wav[0, 0, :])
assert torch.count_nonzero(wav[0, 0, :first_zero_index]) == first_zero_index
assert torch.count_nonzero(wav[0, 0, first_zero_index:]) == 0
receptive_field_secs = first_zero_index / 16_000
print(f"The receptive field seems to be {first_zero_index}, aka {receptive_field_secs} seconds.")
