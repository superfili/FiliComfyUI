from .cd_base64_image import cd_base64_image, cd_image_base64

NODE_CLASS_MAPPINGS = {
    "base64_image": cd_base64_image,
    "image_base64": cd_image_base64,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "base64_image": "Base64 -> Image",
    "image_base64": "Image -> Base64",
}

WEB_DIRECTORY = "./js"

# End registration
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
