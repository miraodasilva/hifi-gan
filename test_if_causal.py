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
melspec = torch.cat([torch.zeros(2, 80, 50), torch.ones(2, 80, 50)], dim=-1).to(device)
wav = generator(melspec)
## Then we check if the first half of the resulting waveform is all zeros, implying that there is not dependence with future inputs
if torch.count_nonzero(wav[:, :, :8000]).item() == 0:
    print("Looks like the network is causal.")
else:
    print("Looks like the network is not causal.")
## PS: This test only works if we turn off the biases in all layers via bias=False, but for the real network we should turn them on as default (bias=True).

assert (
    torch.count_nonzero(wav[:, :, 8000:]).item() == wav.size(0) * wav.size(1) * 8000
)  # make sure the rest of the output isn't zeros, just to be sure
