import wandb
from train import main
import torch
from audiocraft.models import MusicGen
from generate_inf import generate_long_seq
from util import display_audio
import sys

dataset_cfg = dict(
        dataset_path_train = "train",
        dataset_path_eval = "eval",
        batch_size=4,
        num_examples_train= 1000,
        num_examples_eval= 200,
        segment_duration= 30,
        sample_rate= 32_000,
        shuffle= True,
        return_info= False)

cfg = dict(
    learning_rate = 0.0001,
    epochs = 80,
    model = "small",
    seed = (hash("blabliblu") % 2**32 - 1),
    use_wandb = True
)

if __name__ == '__main__':

    if sys.argv[1] == 'train':
        wandb.login()
        main(sys.argv[1], cfg, dataset_cfg, '')

    elif sys.argv[1] == 'generate':
        print('loading model')
        model = MusicGen.get_pretrained('small')
        model.lm.load_state_dict(torch.load(f'./models/lm_{sys.argv[2]}_final.pt'))

        model.set_generation_params(
            use_sampling=True,
            top_k=250,
            duration=30,
        )
        print('generating sequence')
        num_samples = 8
        total_gen_len = 1024
        temp = 1.1
        use_sampling = True
        top_k = 2000
        top_p = 0
        out = generate_long_seq(model, num_samples, total_gen_len, use_sampling, temp, top_k, top_p, None)
        print('writing audio', sys.argv[2])
        display_audio(out, path=f"./generated/{sys.argv[2]}_{total_gen_len}l_{temp}t_{top_k}k_{top_p}p.wav")