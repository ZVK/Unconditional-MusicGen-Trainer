import wandb
from train import main
import torch
from audiocraft.models import MusicGen
from generate_inf import generate_long_seq
from util import display_audio
import sys
#from datetime import datetime

dataset_cfg = dict(
        dataset_path_train = "train",
        dataset_path_eval = "eval",
        batch_size=8,
        num_examples_train= 6000,
        num_examples_eval= 800,
        segment_duration= 30,
        sample_rate= 32_000,
        shuffle= True,
        return_info= False)

try:
    _pt = sys.argv[3]
except:
    _pt = None

cfg = dict(
    learning_rate = 0.0001,
    epochs = 30,
    model = "small",
    seed = (hash("blabliblu") % 2**32 - 1),
    use_wandb = True,
    pt = _pt)

if __name__ == '__main__':
    
    if sys.argv[1] == 'train':
        if cfg['use_wandb']:
            wandb.login()
        #ts = datetime.timestamp(datetime.now())
        main(f'{sys.argv[2]}', cfg, dataset_cfg, '')

    elif sys.argv[1] == 'generate':
        print('loading model')
        model = MusicGen.get_pretrained(cfg['model'])
        model.lm.load_state_dict(torch.load(f'./models/lm_{sys.argv[2]}_final.pt'))

        model.set_generation_params(
            use_sampling=True,
            top_k=100,
            duration=30,
        )
        print('generating sequence')
        num_samples = 8
        total_gen_len = 1024
        temp = 1.01
        use_sampling = True
        top_k = 250
        top_p = 0
        out = generate_long_seq(model, num_samples, total_gen_len, use_sampling, temp, top_k, top_p, None)
        print('writing audio', sys.argv[2])
        display_audio(out, path=f"./generated/{sys.argv[2]}_{total_gen_len}l_{temp}t_{top_k}k_{top_p}p_{num_samples}s.wav")