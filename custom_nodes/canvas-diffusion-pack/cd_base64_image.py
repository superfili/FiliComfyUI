import base64
import hashlib
import numpy as np
import torch
import io

from PIL import Image, ImageSequence, ImageOps


@staticmethod
def decode_image(image64):
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


@staticmethod
def encode_image(image):
    image = image.cpu().detach().numpy()
    image = image * 255.0
    image = image.astype(np.uint8)
    image = image[0]

    image = Image.fromarray(image, "RGB")
    output = io.BytesIO()
    image.save(output, format="PNG")
    image64 = base64.b64encode(output.getvalue()).decode("utf-8")
    return image64


class cd_base64_image:
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "decode"
    CATEGORY = "CanvasDiffusion"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image64": ("STRING", {"multiline": True})}}

    @classmethod
    def decode(s, image64):
        return decode_image(image64)

    @classmethod
    def IS_CHANGED(s, image64):
        m = hashlib.sha256()
        m.update(image64)
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image64):
        return True


class cd_image_base64:
    RETURN_TYPES = ("STRING",)
    FUNCTION = "encode_and_output"
    CATEGORY = "CanvasDiffusion"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "text": ("STRING", {"multiline": True})
            }
        }

    @classmethod
    def encode_and_output(s, text, image):
        b64 = encode_image(image)
        return {
            "result": (b64,),
            "ui": {"text": (b64, )}
        }

    @classmethod
    def IS_CHANGED(s, image):
        m = hashlib.sha256()
        m.update(image.cpu().numpy().tobytes())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        return True
