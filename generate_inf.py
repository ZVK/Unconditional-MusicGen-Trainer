import torch, torchaudio
import numpy as np
import random
from audiocraft.utils.autocast import TorchAutocast
from audiocraft.data.audio_utils import convert_audio
from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout
from train import get_condition_tensor

def seed_continuation(model, prompt, sample_rate):
    prompt = convert_audio(prompt, sample_rate, model.sample_rate, model.audio_channels)

    attributes, _ = model._prepare_tokens_and_attributes(prompt, None)
    descriptions = [None]
    attributes, prompt_tokens = model._prepare_tokens_and_attributes(descriptions, prompt)
    assert prompt_tokens is not None
    return attributes, prompt_tokens

def generate_long_seq(
    model, num_samples, total_gen_len, use_sampling=True, temp=1.0, top_k=250, top_p=0.0, cfg_coef=None, input_audio=None, prompt_duration=2
):
    """Instead of using a text prompt, half the sample length of the previous generation is used as prompt input 

    Args:
        model (MusicGen): The pretrained MusicGen Model
        num_samples (integer): How many samples should be created, one sample is half the totgal_gen_len
        total_gen_len (integer): Maximum generation length.
        use_sampling (bool): Whether to use a sampling strategy or not.
        temp (float): Softmax temperature parameter. Defaults to 1.0.
        top_k (integer): top_k used for sampling. Defaults to 250.
        top_p (float): top_p used for sampling, when set to 0 top_k is used. Defaults to 0.0.
        cfg_coef (float): Coefficient used for classifier free guidance. Defaults to 3.0.

    Returns:
        List[torch.Tensor]: Output is a list of numpy vectors
    """
    samples = []
    generator = set_random_seed()
    if input_audio:
        # seed with audio prompt continuation
        print(f'seeding audio continuation with {input_audio}')
        prompt_waveform, prompt_sr = torchaudio.load(input_audio)
        print(f'loaded waveform with shape {prompt_waveform}')
        prompt_waveform = prompt_waveform[..., :int(prompt_duration * prompt_sr)]
        print('trimmed to {prompt_duration} with new shape {prompt_waveform.shape}')
        attributes, prompt_tokens = seed_continuation(model, prompt_waveform, prompt_sr)
    else:
        # normal start with no prompt
        descriptions = [None]
        attributes, prompt_tokens = model._prepare_tokens_and_attributes(descriptions, None)
    prev_generated = prompt_tokens
    remove_prompts = False
    vs = []
    print('_____________________________________')
    with model.autocast:
        for i in range(num_samples):
            sample_state = generator.get_state()
            sampleL = model.lm.generate(
                prev_generated,
                attributes,
                1,
                total_gen_len,
                use_sampling,
                temp,
                top_k,
                top_p,
                cfg_coef,
                generator=generator,
                remove_prompts=remove_prompts,
            )
            generator.set_state(sample_state)
            sampleR = model.lm.generate(
                prev_generated,
                attributes,
                1,
                total_gen_len,
                use_sampling,
                temp,
                top_k,
                top_p,
                cfg_coef,
                generator=generator,
                remove_prompts=remove_prompts,
            )
            generator.set_state(sample_state)
            sampleL2 = model.lm.generate(
                prev_generated,
                attributes,
                1,
                total_gen_len,
                use_sampling,
                temp+((1.-temp)*0.5),
                top_k,
                top_p,
                cfg_coef,
                generator=generator,
                remove_prompts=remove_prompts,
            )
            generator.set_state(sample_state)
            sampleR2 = model.lm.generate(
                prev_generated,
                attributes,
                1,
                total_gen_len,
                use_sampling,
                temp+((1.-temp)*0.5),
                top_k,
                top_p,
                cfg_coef,
                generator=generator,
                remove_prompts=remove_prompts,
            )
            generator.set_state(sample_state)
            sampleC = model.lm.generate(
                prev_generated,
                attributes,
                1,
                total_gen_len,
                use_sampling,
                temp+((1.-temp)*0.5),
                0,
                top_p,
                cfg_coef,
                generator=generator,
                remove_prompts=remove_prompts,
            )
            sample_stack = torch.cat([torch.Tensor(sampleC).float(),
                                      torch.Tensor(sampleL).float(),
                                      torch.Tensor(sampleR).float(),
                                      torch.Tensor(sampleL2).float(),
                                      torch.Tensor(sampleR2).float()])
            # model pooling statistics
            median, _ = torch.median(sample_stack, dim=0, keepdim=True)
            mode, _ = torch.mode(sample_stack, dim=0, keepdim=True)
            std = torch.std(sample_stack, dim=0, keepdim=True)
            # range of standard deviation
            std_min = torch.min(std)
            std_max = torch.max(std)
            #print(std.shape, median.shape, mode.shape)
            std_thresh = (((std_max - std_min) * 0.5) + std_min)
            #print(f'STD: mean {std_thresh} | min {std_min} | max {std_max}')
            vs.append(torch.sum(torch.sum(std, dim=1)).detach().cpu().numpy())
            median_std_masks = torch.squeeze_copy(torch.where(std > std_thresh, torch.zeros_like(std), torch.ones_like(std)))
            #print(median_std_masks)
            for i in range(median_std_masks.shape[0]):
                statmap = f'\n{i}|'
                for s in median_std_masks[i-1]:
                    if s == 0:
                        statmap += '░'
                    elif s == 1:
                        statmap += '█'
                print(statmap)
            sample = torch.where(std > std_thresh, median, mode)
            sample = sample.int()
            if sample.shape[-1] == 1024:
                print(f'\t▄▄█{sample.shape[-1]}▀▀█')
            elif sample.shape[-1] == 512:
                print(f'\t▄█▀{sample.shape[-1]}▀█▄')
            elif sample.shape[-1] == 256:
                print(f'\t▄█{sample.shape[-1]}▀█')
            elif sample.shape[-1] == 128:
                print(f'\t█{sample.shape[-1]}▄▄')
            elif sample.shape[-1] == 64:
                print(f'\t▀{sample.shape[-1]}▄')
            else:
                print(f'\t░░{sample.shape[-1]}░░')
            remove_prompts = True
            
            if sample.shape[2] == total_gen_len:
                prev_generated = torch.clone(sample[..., total_gen_len // 2 :])
            else:
                prev_generated = torch.clone(sample)

            with torch.no_grad():
                gen_audioL = model.compression_model.decode(sampleL, None)
                gen_audioR = model.compression_model.decode(sampleR, None)
                gen_audioL2 = model.compression_model.decode(sampleL2, None)
                gen_audioR2 = model.compression_model.decode(sampleR2, None)
                gen_audioC = model.compression_model.decode(sampleC, None)
            # free gpu
            del sampleL, sampleR, sampleL2, sampleR2, sampleC

            gen_audioL = gen_audioL[0].detach().cpu().numpy()
            gen_audioL = gen_audioL.transpose(1, 0)
            gen_audioR = gen_audioR[0].detach().cpu().numpy()
            gen_audioR = gen_audioR.transpose(1, 0)
            gen_audioL2 = gen_audioL2[0].detach().cpu().numpy()
            gen_audioL2 = gen_audioL2.transpose(1, 0)
            gen_audioR2 = gen_audioR2[0].detach().cpu().numpy()
            gen_audioR2 = gen_audioR2.transpose(1, 0)
            gen_audioC = gen_audioC[0].detach().cpu().numpy()
            gen_audioC = gen_audioC.transpose(1, 0)

            # stereo blend
            L = (gen_audioL * 0.4) + (gen_audioL2 * 0.3) + (gen_audioC * 0.15) + (gen_audioR2 * 0.15)
            R = (gen_audioR * 0.4) + (gen_audioR2 * 0.3) + (gen_audioC * 0.15) + (gen_audioL2 * 0.15)
            samples.append([L, R])
    print(f'summed standard deviation for all channels in audio chunk {np.stack(vs)/np.max(vs)}')
    return samples

def set_random_seed(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator(device='cuda').manual_seed(seed)
    np.random.seed(seed)
    return generator

def generate_long_seq_ensemble(
    model, models, num_samples, total_gen_len, use_sampling=True, temp=1.0, top_k=250, top_p=0.0, cfg_coef=None, input_audio=None, prompt_duration=256
):
    """Using an ensemble of snapshot models: https://arxiv.org/pdf/1704.00109.pdf
    Instead of using a text prompt, half the sample length of the previous generation is used as prompt input 

    Args:
        model (MusicGen): The pretrained MusicGen Model
        models (list): A list of trained model file paths to ensemble
        num_samples (integer): How many samples should be created, one sample is half the totgal_gen_len
        total_gen_len (integer): Maximum generation length.
        use_sampling (bool): Whether to use a sampling strategy or not.
        temp (float): Softmax temperature parameter. Defaults to 1.0.
        top_k (integer): top_k used for sampling. Defaults to 250.
        top_p (float): top_p used for sampling, when set to 0 top_k is used. Defaults to 0.0.
        cfg_coef (float): Coefficient used for classifier free guidance. Defaults to 3.0.

    Returns:
        List[torch.Tensor]: Output is a list of numpy vectors
    """
    samples = []
    generator = set_random_seed()
    if input_audio:
        # seed with audio prompt continuation
        print(f'seeding audio continuation with {input_audio}')
        prompt_waveform, prompt_sr = torchaudio.load(input_audio)
        print(f'loaded waveform with shape {prompt_waveform}')
        prompt_waveform = prompt_waveform[..., :int(prompt_duration * prompt_sr)]
        print('trimmed to {prompt_duration} with new shape {prompt_waveform.shape}')
        attributes, prompt_tokens = seed_continuation(model, prompt_waveform, prompt_sr)
    else:
        # normal start with no prompt
        descriptions = ["Metalcore"]
        attributes, prompt_tokens = model._prepare_tokens_and_attributes(descriptions, None)
        #print(len(attributes), attributes)
    prev_generated = prompt_tokens
    remove_prompts = False
    with model.autocast:
        for i in range(num_samples):
            m_samples = []
            sample_state = generator.get_state()
            for n, model_path in enumerate(models):
                if n > 0:
                    generator.set_state(sample_state)
                print(f'loading {model_path} {n+1} of {len(models)}')
                model.lm.load_state_dict(torch.load(model_path))
                #model.set_generation_params(
                #        use_sampling=True,
                #        top_k=250, # neighbors
                #        duration=30, # seconds
                #        )
                _sample = model.lm.generate(
                    prev_generated,
                    attributes,
                    1,
                    total_gen_len,
                    use_sampling,
                    temp,
                    top_k,
                    top_p,
                    cfg_coef,
                    generator=generator,
                    remove_prompts=remove_prompts,
                )
                m_samples.append(torch.Tensor(_sample).float())
            sample_stack = torch.cat(m_samples)
            assert(sample_stack.shape[0] == len(models))
            # median pooling
            median, _ = torch.median(sample_stack, dim=0, keepdim=True)
            mode, _ = torch.mode(sample_stack, dim=0, keepdim=True)
            # strategy to select mode vs median based on mid std
            std = torch.std(sample_stack, dim=0, keepdim=True)
            print(torch.sum(torch.sum(std, dim=1)))
            print(std.shape)
            std_min = torch.min(std)
            std_max = torch.max(std)
            std_thresh = (((std_max - std_min) * 0.5) + std_min)
            print(f'STD: mean {std_thresh} | min {std_min} | max {std_max}')
            sample = torch.where(std > std_thresh, median, mode)
            print(sample.shape)

            remove_prompts = True

            if sample.shape[2] == total_gen_len:
                prev_generated = torch.clone(sample[..., total_gen_len // 2 :])
            else:
                prev_generated = torch.clone(sample)
            print(f'decoding')
            with torch.no_grad():
                # use the most recent decoder
                gen_audioL = model.compression_model.decode(sample.int(), None)
                # use the oldest decoder
                model.lm.load_state_dict(torch.load(models[0])) 
                gen_audioR = model.compression_model.decode(sample.int(), None)
            # free gpu
            del sample

            gen_audioL = gen_audioL[0].detach().cpu().numpy()
            gen_audioL = gen_audioL.transpose(1, 0)
            gen_audioR = gen_audioR[0].detach().cpu().numpy()
            gen_audioR = gen_audioR.transpose(1, 0)
            L = (gen_audioL * 0.66) * (gen_audioR * 0.33)
            R = (gen_audioR * 0.66) * (gen_audioL * 0.33)
            samples.append([L, R])

    return samples