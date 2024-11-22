import cv2
import einops
import numpy as np
import torch
import random
from glob import glob
from tqdm import tqdm
from datetime import datetime
import os
import argparse
import json

from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity
disable_verbosity()

model = create_model('models/attentionhand.yaml').cpu()
model_path = 'weights/attentionhand.ckpt'
model.load_state_dict(load_state_dict(model_path, location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_path', type=str, dest='sample_path', default='samples/modalities.json')
    args = parser.parse_args()
    return args

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def process(input_image, prompt, a_prompt='', n_prompt='', num_samples=4, image_resolution=512, ddim_steps=20, guess_mode=False, strength=1.0, scale=9.0, seed=42, eta=0.0):
    
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        control = torch.from_numpy(img.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    
    return results

if __name__ == "__main__":
    pairs = []
    args = parse_args()
    with open(args.sample_path, 'rt') as f:
        for line in f:
            pairs.append(json.loads(line))
    f.close()
    
    for pair_id, pair in enumerate(pairs):
        mesh_path = pair['mesh']
        text_path = pair['text']
        mesh = cv2.imread(mesh_path, cv2.IMREAD_GRAYSCALE)
        mesh = cv2.resize(mesh, (512, 512))
        with open(text_path, 'r') as f:
            text = f.readline()
        f.close()
        results = process(mesh, text)
        for result_id, result in enumerate(results):
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join('result_' + str(pair_id).zfill(3) + '_' + str(result_id) + '.png'), result)

    print("Finished.")

    # a_prompt = ''
    # n_prompt = ''
    # num_samples = 4
    # image_resolution = 512
    # ddim_steps = 20
    # strength = 1.0
    # scale = 9.0
    # seed = 42
    # eta = 0.0
    # guess_mode = False
    
    # texts = [
    #     'A woman sitting in front of a laptop computer.',
    #     'A boy in a plaid shirt holding an umbrella.',
    #     'A man holding the bridles of a horse.',
    #     'Man in mid bite of pizza with wall behind.',
    #     'A man holding a camera up over his left shoulder.',
    #     'A man on a boat holding his dog in his lap.',
    #     'A man kneeling down next to a brown dog.',
    #     'A tray topped with bowls of different kinds of food.',
    #     'A man is holding skis and ski poles.',
    #     'A man cutting up a large sandwich in a kitchen.',
    #     'A man with eyeglasses working on a laptop computer.',
    #     'A young boy standing next to a fire hydrant on green grass.',
    #     'A man in a hat is holding two cell phones.',
    #     'A man and woman standing together talking.',
    #     'A bearded man in a red cap on a skateboard.',
    #     'A man taking on a cell phone. The man is wearing a formal outfit.',
    #     'A tennis player in action holding his racket.',
    #     'A young man pauses while eating a sandwich.',
    #     'A boy is playing video games in his bedroom.',
    #     'A man reading something on his cell phone.',
    #     'A kid eating a doughnut at a table.',
    #     'A man using a laptop in a cafe.',
    #     'A young man in a kitchen shapes dough into balls.',
    #     'A man with headphones around his neck and holding a skateboard.'
    # ]
    # mesh_paths = sorted(glob('/database2/hand/backup/eccv/mscoco/images_render_bighand_upgrade/train2017/*'))
    # for mesh_path in tqdm(mesh_paths, desc='mesh'):
    #     for text_id, text in enumerate(tqdm(texts, desc='text')):
    #         mesh_id = mesh_path.split('/')[-1][:-4]
    #         save_folder = os.path.join('result', mesh_id, 'text_' + str(text_id).zfill(2))
    #         os.makedirs(save_folder, exist_ok=True)
    #         mesh_img = cv2.imread(mesh_path, cv2.IMREAD_GRAYSCALE)
    #         mesh_img = cv2.resize(mesh_img, (512, 512))
    #         cv2.imwrite(os.path.join(save_folder, 'mesh.png'), mesh_img)
    #         with open(os.path.join(save_folder, 'text.txt'), 'w') as f:
    #             f.write(text)
    #         f.close()
    #         results = process(mesh_img, text, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)

    #         for result_id, result in enumerate(results):
    #             result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    #             cv2.imwrite(os.path.join(save_folder, 'result_' + str(result_id) + '.png'), result)
    
    # print("Finished.")