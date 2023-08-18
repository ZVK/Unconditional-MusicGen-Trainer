import torch
import numpy as np
import IPython.display as ipd
import typing as tp

from scipy.io.wavfile import write


def display_audio(samples: tp.List[tp.List[torch.Tensor]], path: str = None):
    left_channel = np.concatenate([sample[0] for sample in samples], axis=0)
    right_channel = np.concatenate([sample[1] for sample in samples], axis=0)
    
    stereo_audio = np.squeeze(np.stack((left_channel, right_channel), axis=-1))
    #ipd.display(ipd.Audio(stereo_audio, rate=32000))
    print(stereo_audio.shape)
    if path:
        write(path, 32000, stereo_audio.astype(np.float32))
