import os,sys
import PIL
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as tvtrans
from .lib.cfg_helper import model_cfg_bank
from .lib.model_zoo import get_model
from random import randint
from .lib.model_zoo.ddim import DDIMSampler
      

def highlight_print(info):
    print('')
    print(''.join(['#']*(len(info)+4)))
    print('# '+info+' #')
    print(''.join(['#']*(len(info)+4)))
    print('')

def decompose(x, q=20, niter=100):
    x_mean = x.mean(-1, keepdim=True)
    x_input = x - x_mean
    u, s, v = torch.pca_lowrank(x_input, q=q, center=False, niter=niter)
    ss = torch.stack([torch.diag(si) for si in s])
    x_lowrank = torch.bmm(torch.bmm(u, ss), torch.permute(v, [0, 2, 1]))
    x_remain = x_input - x_lowrank
    return u, s, v, x_mean, x_remain

class adjust_rank(object):
    def __init__(self, max_drop_rank=[1, 5], q=20):
        self.max_semantic_drop_rank = max_drop_rank[0]
        self.max_style_drop_rank = max_drop_rank[1]
        self.q = q

        def t2y0_semf_wrapper(t0, y00, t1, y01):
            return lambda t: (np.exp((t-0.5)*2)-t0)/(t1-t0)*(y01-y00)+y00
        t0, y00 = np.exp((0  -0.5)*2), -self.max_semantic_drop_rank
        t1, y01 = np.exp((0.5-0.5)*2), 1
        self.t2y0_semf = t2y0_semf_wrapper(t0, y00, t1, y01)

        def x2y_semf_wrapper(x0, x1, y1):
            return lambda x, y0: (x-x0)/(x1-x0)*(y1-y0)+y0
        x0 = 0
        x1, y1 = self.max_semantic_drop_rank+1, 1
        self.x2y_semf = x2y_semf_wrapper(x0, x1, y1)
        
        def t2y0_styf_wrapper(t0, y00, t1, y01):
            return lambda t: (np.exp((t-0.5)*2)-t0)/(t1-t0)*(y01-y00)+y00
        t0, y00 = np.exp((1  -0.5)*2), -(q-self.max_style_drop_rank)
        t1, y01 = np.exp((0.5-0.5)*2), 1
        self.t2y0_styf = t2y0_styf_wrapper(t0, y00, t1, y01)

        def x2y_styf_wrapper(x0, x1, y1):
            return lambda x, y0: (x-x0)/(x1-x0)*(y1-y0)+y0
        x0 = q-1
        x1, y1 = self.max_style_drop_rank-1, 1
        self.x2y_styf = x2y_styf_wrapper(x0, x1, y1)

    def __call__(self, x, lvl):
        if lvl == 0.5:
            return x

        if x.dtype == torch.float16:
            fp16 = True
            x = x.float()
        else:
            fp16 = False
        std_save = x.std(axis=[-2, -1])

        u, s, v, x_mean, x_remain = decompose(x, q=self.q)

        if lvl < 0.5:
            assert lvl>=0
            for xi in range(0, self.max_semantic_drop_rank+1):
                y0 = self.t2y0_semf(lvl)
                yi = self.x2y_semf(xi, y0)
                yi = 0 if yi<0 else yi
                s[:, xi] *= yi

        elif lvl > 0.5:
            assert lvl <= 1
            for xi in range(self.max_style_drop_rank, self.q):
                y0 = self.t2y0_styf(lvl)
                yi = self.x2y_styf(xi, y0)
                yi = 0 if yi<0 else yi
                s[:, xi] *= yi
            x_remain = 0

        ss = torch.stack([torch.diag(si) for si in s])
        x_lowrank = torch.bmm(torch.bmm(u, ss), torch.permute(v, [0, 2, 1]))
        x_new = x_lowrank + x_mean + x_remain

        std_new = x_new.std(axis=[-2, -1])
        x_new = x_new / std_new * std_save

        if fp16:
            x_new = x_new.half()

        return x_new

class Reconstructor(object):
    def __init__(self, fp16=True, device="cuda:0", cache_dir="../cache", ddim_steps=50, deprecated=False):
        print("Reconstructor: Loading model... fp16: ", fp16)
        if deprecated:
            cfgm_name = 'vd_noema'
        cfgm_name = 'vd_four_flow_v1-0'
        
        cfgm = model_cfg_bank()(cfgm_name)
        cfgm['args']['vae_cfg_list'][0][1]['pth'] = f'{cache_dir}/kl-f8.pth'
        cfgm['args']['vae_cfg_list'][1][1]['pth'] =f'{cache_dir}/optimus-vae.pth'
        net = get_model()(cfgm)

        if fp16:
            net.ctx['text'].fp16 = True
            net.ctx['image'].fp16 = True
            net = net.half()
            self.dtype = torch.float16
            if deprecated:
                sd = torch.load(f'{cache_dir}/vd-four-flow-v1-0-fp16-deprecated.pth', map_location='cpu')
            else:
                sd = torch.load(f'{cache_dir}/vd-four-flow-v1-0-fp16.pth', map_location='cpu')
        else:
            self.dtype = torch.float32
            sd = torch.load(f'{cache_dir}/vd-four-flow-v1-0.pth', map_location='cpu')
                
            # from huggingface_hub import hf_hub_download
            # if fp16:
            #     temppath = hf_hub_download('shi-labs/versatile-diffusion-model', 'pretrained_pth/vd-four-flow-v1-0-fp16.pth')
            # else:
            #     temppath = hf_hub_download('shi-labs/versatile-diffusion-model', 'pretrained_pth/vd-four-flow-v1-0.pth')
            # sd = torch.load(temppath, map_location='cpu')

        net.load_state_dict(sd, strict=False)

        self.device=device
        net.to(self.device)
        self.net = net
        self.sampler = DDIMSampler(net)

        self.output_dim = [512, 512]

        self.ddim_steps = ddim_steps
        self.ddim_eta = 0.0
        self.image_latent_dim = 4
        self.sampler.make_schedule(ddim_num_steps=self.ddim_steps, ddim_eta=self.ddim_eta, verbose=False)
        self.adjust_rank_f = adjust_rank(max_drop_rank=[1, 5], q=20)
        self.scale = 7.5
        self.disentanglement_noglobal = True
    def embed_text(self, prompt):
        if isinstance(prompt, str):
            prompt = [prompt]
        text_encoding = self.net.ctx_encode(prompt, which='text')
        return text_encoding
    
    def embed_image(self, image):
        if isinstance(image, PIL.Image.Image):
            image = tvtrans.ToTensor()(image)
        image = tvtrans.Resize([512, 512], interpolation=PIL.Image.BICUBIC)(image)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device).to(self.dtype)
        image_encoding = self.net.ctx_encode(image, which='image')
        return image_encoding
    
    def project_clip(self, expanded_clip):
        reduced_clip = expanded_clip[:, 0, :]
        reduced_clip = reduced_clip * torch.norm(reduced_clip, dim=-1, keepdim=True)
        print("RECONSTRUCTOR REDUCED CLIP SHAPE: ", reduced_clip.shape)
        projected_clip = self.net.ctx["image"].model.visual_projection(reduced_clip)
        return projected_clip
    
    def reconstruct(self, 
                    image=None, 
                    c_i=None, 
                    c_t=None, 
                    n_samples=1, 
                    textstrength=0.5, 
                    strength=1.0, 
                    color_adjust=False,
                    fcs_lvl=0.5, 
                    seed=None
                    ):
        
        numClips =0
        h, w = 512, 512
        BICUBIC = PIL.Image.Resampling.BICUBIC
        
        if strength == 0:
            return [image]*n_samples
        else:
            assert (c_t is not None) or (c_i is not None)
            c_info_list = []
            scale = self.scale
            if c_t is not None and textstrength != 0:
                c_t = c_t.reshape((77,768)).to(dtype=torch.float16, device=self.device)
                ut = self.net.ctx_encode([""], which='text').repeat(n_samples, 1, 1)
                ct = c_t.repeat(n_samples, 1, 1)
                c_info_list.append({
                    'type':'text', 
                    'conditioning':ct.to(torch.float16), 
                    'unconditional_conditioning':ut,
                    'unconditional_guidance_scale':scale,
                    'ratio': textstrength, })
                numClips +=1
            else:
                textstrength=0

            if c_i is not None and textstrength != 1:
                c_i = c_i.reshape((257,768)).to(dtype=torch.float16, device=self.device)
                ci = c_i

                if self.disentanglement_noglobal:
                    ci_glb = ci[:, 0:1]
                    ci_loc = ci[:, 1: ]
                    ci_loc = self.adjust_rank_f(ci_loc, fcs_lvl)
                    ci = torch.cat([ci_glb, ci_loc], dim=1).repeat(n_samples, 1, 1)
                else:
                    ci = self.adjust_rank_f(ci, fcs_lvl).repeat(n_samples, 1, 1)

                c_info_list.append({
                    'type':'image', 
                    'conditioning':ci.to(torch.float16), 
                    'unconditional_conditioning':torch.zeros_like(ci),
                    'unconditional_guidance_scale':scale,
                    'ratio': (1-textstrength), })
                numClips +=1
            else:
                textstrength=1
        if(image is not None):
            image_tensor = tvtrans.Compose([
                tvtrans.ToTensor(),
                tvtrans.Resize((w, h))
            ])(image).to(self.device).to(self.dtype)
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
        shape = [n_samples, self.image_latent_dim, h//8, w//8]
        if(seed):
            np.random.seed(seed)
            torch.manual_seed(seed + 100)
        else:
            seed = randint(0,1000)
            np.random.seed(seed)
            torch.manual_seed(seed + 100)
        if strength!=1 and image:
            x0 = self.net.vae_encode(image_tensor, which='image').repeat(n_samples, 1, 1, 1)
            step = int(self.ddim_steps * (strength))
            if numClips==2:
                x, _ = self.sampler.sample_multicontext(
                    steps=self.ddim_steps,
                    x_info={'type':'image', 'x0':x0, 'x0_forward_timesteps':step},
                    c_info_list=c_info_list,
                    shape=shape,
                    verbose=False,
                    eta=self.ddim_eta)
            else:
                x, _ = self.sampler.sample(
                    steps=self.ddim_steps,
                    x_info={'type':'image', 'x0':x0, 'x0_forward_timesteps':step},
                    c_info=c_info_list[0],
                    shape=shape,
                    verbose=False,
                    eta=self.ddim_eta)
        else:
            if numClips ==2:
                x, _ = self.sampler.sample_multicontext(
                    steps=self.ddim_steps,
                    x_info={'type':'image',},
                    c_info_list=c_info_list,
                    shape=shape,
                    verbose=False,
                    eta=self.ddim_eta)
            else:
                x, _ = self.sampler.sample(
                    steps=self.ddim_steps,
                    x_info={'type':'image',},
                    c_info=c_info_list[0],
                    shape=shape,
                    verbose=False,
                    eta=self.ddim_eta)
        imout = self.net.vae_decode(x, which='image')
        if color_adjust:
            cx_mean = image_tensor.view(3, -1).mean(-1)[:, None, None]
            cx_std  = image_tensor.view(3, -1).std(-1)[:, None, None]
            imout_mean = [imouti.view(3, -1).mean(-1)[:, None, None] for imouti in imout]
            imout_std  = [imouti.view(3, -1).std(-1)[:, None, None] for imouti in imout]
            imout = [(ii-mi)/si*cx_std+cx_mean for ii, mi, si in zip(imout, imout_mean, imout_std)]
            imout = [torch.clamp(ii, 0, 1) for ii in imout]
        imout = [tvtrans.ToPILImage()(i) for i in imout]
        if len(imout)==1:
            return imout[0]
        else:
            return imout