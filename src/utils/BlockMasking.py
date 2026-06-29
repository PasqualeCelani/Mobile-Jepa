import math
from multiprocessing import Value
import torch

class MaskCollator(object):

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.15, 0.25),
        pred_mask_scale=(0.1, 0.15),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=10,
        allow_overlap=False
    ):
        super(MaskCollator, self).__init__()

        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2

        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size

        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self._itr_counter = Value('i', -1)

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()

        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)

        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))

        h = max(1, h)
        w = max(1, w)

        while h >= self.height: h -= 1
        while w >= self.width: w -= 1

        return (h, w)

    def __call__(self, batch):
        B = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)
        
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))
            
        e_h, e_w = e_size
        p_h, p_w = p_size
        

        max_e_top = max(0, self.height - e_h)
        max_e_left = max(0, self.width - e_w)
        
        enc_tops = torch.randint(0, max_e_top + 1, (B, self.nenc), generator=g)
        enc_lefts = torch.randint(0, max_e_left + 1, (B, self.nenc), generator=g)
        

        max_p_top = max(0, self.height - p_h)
        max_p_left = max(0, self.width - p_w)
        

        if self.allow_overlap or (p_h * p_w < self.min_keep):
            pred_tops = torch.randint(0, max_p_top + 1, (B, self.npred))
            pred_lefts = torch.randint(0, max_p_left + 1, (B, self.npred))
        else:
            valid_preds = torch.zeros((B, self.npred), dtype=torch.bool)
            cand_tops = torch.randint(0, max_p_top + 1, (B, self.npred))
            cand_lefts = torch.randint(0, max_p_left + 1, (B, self.npred))
            
            attempts = 0
            MAX_ATTEMPTS = 120 
            
            while not valid_preds.all() and attempts < MAX_ATTEMPTS:
                invalid_mask = ~valid_preds
                inv_b, inv_p = torch.where(invalid_mask)
                
                if len(inv_b) == 0:
                    break
                    

                enc_t = enc_tops[inv_b] 
                enc_l = enc_lefts[inv_b] 
                
                c_t = cand_tops[inv_b, inv_p].unsqueeze(1) 
                c_l = cand_lefts[inv_b, inv_p].unsqueeze(1) 
                
                no_overlap = (
                    (c_t + p_h <= enc_t) |
                    (enc_t + e_h <= c_t) |
                    (c_l + p_w <= enc_l) |
                    (enc_l + e_w <= c_l)
                )
                
                is_valid = no_overlap.all(dim=1)
                
                valid_preds[inv_b, inv_p] = is_valid
                
                still_invalid = ~is_valid
                if still_invalid.any():
                    si_b = inv_b[still_invalid]
                    si_p = inv_p[still_invalid]
                    
                    new_t = torch.randint(0, max_p_top + 1, (int(still_invalid.sum()),))
                    new_l = torch.randint(0, max_p_left + 1, (int(still_invalid.sum()),))
                    
                    cand_tops[si_b, si_p] = new_t
                    cand_lefts[si_b, si_p] = new_l
                    
                attempts += 1
                
            pred_tops = cand_tops
            pred_lefts = cand_lefts

        enc_bboxes = torch.stack([
            enc_tops * self.patch_size,
            enc_lefts * self.patch_size,
            torch.full_like(enc_tops, e_h * self.patch_size),
            torch.full_like(enc_tops, e_w * self.patch_size)
        ], dim=-1).float()
        
        pred_bboxes = torch.stack([
            pred_tops * self.patch_size,
            pred_lefts * self.patch_size,
            torch.full_like(pred_tops, p_h * self.patch_size),
            torch.full_like(pred_tops, p_w * self.patch_size)
        ], dim=-1).float()

        return collated_batch, enc_bboxes, pred_bboxes