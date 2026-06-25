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
    
    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size 

        tries = 0
        timeout = og_timeout = 20
        valid_mask = False

        while not valid_mask:
            if self.height - h <= 0 or self.width - w <= 0:
                top, left = 0, 0
            else:
                top = torch.randint(0, self.height - h + 1, (1,)).item()
                left = torch.randint(0, self.width - w + 1, (1,)).item()
            
            mask_2d = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask_2d[top:top+h, left:left+w] = 1

            overlap = False
            if acceptable_regions is not None:
                for region in acceptable_regions:
                    if torch.sum(mask_2d * region) > 0:
                        overlap = True
                        break
            

            if overlap:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    if tries > 5: 
                        break
                continue
            
            if h * w >= self.min_keep:
                valid_mask = True
                break
            else:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    if tries > 5:
                        break

        top_px = top * self.patch_size
        left_px = left * self.patch_size
        h_px = h * self.patch_size
        w_px = w * self.patch_size
        
        bbox = (top_px, left_px, h_px, w_px)
        
        return bbox, mask_2d

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

        collated_masks_pred, collated_masks_enc = [], []

        for _ in range(B):
            masks_e = []
            context_2d_masks = []
            for _ in range(self.nenc):
                bbox_e, mask_2d_e = self._sample_block_mask(e_size)
                masks_e.append(bbox_e)
                context_2d_masks.append(mask_2d_e)
            
            collated_masks_enc.append(masks_e)

            masks_p = []
            for _ in range(self.npred):
                unacceptable = context_2d_masks if not self.allow_overlap else None
                bbox_p, _ = self._sample_block_mask(p_size, acceptable_regions=unacceptable)
                masks_p.append(bbox_p)
                
            collated_masks_pred.append(masks_p)

        masks_pred_tensor = torch.tensor(collated_masks_pred, dtype=torch.float32)
        masks_enc_tensor = torch.tensor(collated_masks_enc, dtype=torch.float32)

        return collated_batch, masks_enc_tensor, masks_pred_tensor