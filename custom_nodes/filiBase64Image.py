import base64
import hashlib
import numpy as np
import torch
import io

from PIL import Image, ImageSequence, ImageOps

class FiliBase64Image:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image64": ("STRING", {"multiline": True})}}

    CATEGORY = "fili"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "decode_image"

    def decode_image(self, image64):
        imageBytes = base64.b64decode(image64)
        img = Image.open(io.BytesIO(imageBytes))

        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, image64):
        m = hashlib.sha256()
        m.update(image64)
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image64):
        return True

NODE_CLASS_MAPPINGS = {
    "Fili Base64 Image": FiliBase64Image,
}
