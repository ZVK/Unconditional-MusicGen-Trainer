import wandb
from train import main
import torch
from audiocraft.models import MusicGen
from generate_inf import generate_long_seq, generate_long_seq_ensemble
from util import display_audio
import sys, os.path
from glob import glob

dataset_cfg = dict(
        dataset_path_train = "train",
        dataset_path_eval = "eval",
        batch_size=4,
        num_examples_train= 2000,
        num_examples_eval= 100,
        segment_duration= 30,
        sample_rate= 32_000,
        shuffle= True,
        return_info= False)



if __name__ == '__main__':
    
    if sys.argv[1] == 'train' or sys.argv[1] == 'tune':
        try:
            _pt = sys.argv[3]
        except:
            _pt = None

        cfg = dict(
            learning_rate = 1e-4,
            epochs = 300,
            model = "small",
            seed = (hash("blabliblu") % 2**32 - 1),
            use_wandb = True,
            pt = _pt)
        if sys.argv[1] == 'tune':
            cfg['learning_rate'] = 1e-5
            cfg['epochs'] = 50
        if cfg['use_wandb']:
            wandb.login()
        main(f'{sys.argv[2]}', cfg, dataset_cfg, '')

    elif sys.argv[1] == 'generate' or sys.argv[1] == 'explore' or sys.argv[1] == 'blend':
        cfg = dict(
            model = "small",
            seed = (hash("blabliblu") % 2**32 - 1))
        try:
            input_audio = sys.argv[3]
        except:
            input_audio = None
        audio_duration = 2 # seconds
        print('get pretrained model')
        model = MusicGen.get_pretrained(cfg['model'])
        if sys.argv[1] == 'blend':
            print('loading snapshot ensemble list')
            glob_pattern = os.path.join(sys.argv[2], '*.pt')
            # sort models by creation time
            models = sorted(glob(glob_pattern), key=os.path.getctime, reverse=True)
            assert len(models) > 0
        else:
            print('loading fine tuned state dictionary')
            model.lm.load_state_dict(torch.load(sys.argv[2]))
        #    model.set_generation_params(
        #    use_sampling=True,
        #    top_k=250, # neighbors
        #    duration=30, # seconds
        #)
        use_sampling = True # use sampling strategy
        if sys.argv[1] == 'explore':
            num_sequences = 2 # number of audio chunks to generate in sequence
            # hp search steps
            search_chunks = 2 
            search_tgl = 3
            search_tmp = 5
            search_tk = 4
            search_cfg_coef = 4
        else:
            num_sequences = 1 # number of audio chunks to generate in sequence
            # hp search steps
            search_chunks = 1
            search_tgl = 1
            search_cfg_coef = 4
            if sys.argv[1] == 'blend':
                search_tmp = 1
                search_tk = 1
            else:
                search_tmp = 4
                search_tk = 2

        
        top_p = 0 # {0,1} float to use in sampling as alternative to top_k number
        hps = search_chunks*search_tgl*search_tmp*search_tk
        print(f'starting grid search through {hps} hyperparameter settings')
        for chunk in range(search_chunks): # search audio chunk sizes
            num_chunks = 2**(chunk+1)
            print(f'chunk {num_chunks} {chunk}/{search_chunks}')
            for tgl in range(search_tgl): # search chunk lengths
                total_gen_len = 2**(9+tgl)
                print(f'total gen len {total_gen_len} {tgl}/{search_tgl}')
                for cc in range(search_cfg_coef): # search classifer free guidance coefficient
                    cfg_coef = (3.0 * ((cc + 1) / search_cfg_coef)) + 1.5 # low to high
                    print(f'classifier free guidance coef {cfg_coef}')
                    for t in range(search_tmp): # search temperatures
                        if sys.argv[1] == 'blend':
                            temp = 1.0
                        else:
                            temp = 1.0+(((search_tmp*0.5)-t)*-0.1)
                        print(f'temp {temp} {t}/{search_tmp}')
                        for k in range(search_tk): # search top K neighbors sampling strategies
                            if sys.argv[1] == 'blend':
                                y = 7
                            else:
                                y = 7
                            top_k = 2**(y+k)
                            print(f'total search top k {top_k} {k}/{search_tk}')
                            for sequence in range(num_sequences): # search number of sequences to render with this set of hyper-parameters
                                print(f'generating sequence {sequence+1}')
                                if sys.argv[1] == 'blend':
                                    out = generate_long_seq_ensemble(model, 
                                                            models,
                                                            num_chunks,  
                                                            total_gen_len, 
                                                            use_sampling=use_sampling, 
                                                            temp=temp, 
                                                            top_k=top_k, 
                                                            top_p=top_p, 
                                                            cfg_coef=cfg_coef,
                                                            input_audio=None, 
                                                            prompt_duration=audio_duration)
                                else:
                                    out = generate_long_seq(model, 
                                                            num_chunks, 
                                                            total_gen_len, 
                                                            use_sampling=use_sampling, 
                                                            temp=temp, 
                                                            top_k=top_k, 
                                                            top_p=top_p, 
                                                            cfg_coef=cfg_coef,
                                                            input_audio=None, 
                                                            prompt_duration=audio_duration)

                                print('writing audio', sys.argv[2])
                                hp_string = f"{total_gen_len}x{num_chunks}n_{temp}t_{top_k}k_{top_p}p_{sequence+1}s_{cfg_coef}c"
                                if sys.argv[1] == 'blend':
                                    f_path = f"./generated/{sys.argv[2].replace('/', '-')}_{hp_string}.wav"
                                else:
                                    f_path = f"./generated/{sys.argv[2].split('/')[-1].replace('.pt', '')}_{hp_string}.wav"
                                display_audio(out, path=f_path)
    else:
        print(f'{sys.argv[1]} not implemented') 