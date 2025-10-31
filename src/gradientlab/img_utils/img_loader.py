from PIL import Image
from io import BytesIO
import base64
import requests
from torchvision import transforms

class ImageLoader:
    @staticmethod
    def to_pil(source):
        if isinstance(source, Image.Image):
            return source

        if isinstance(source, str) and source.strip().startswith(("iVBOR", "/9j/", "R0lG", "UklG", "data:image")):
            if source.startswith("data:image"):
                source = source.split(",", 1)[-1]
            img_bytes = base64.b64decode(source)
            return Image.open(BytesIO(img_bytes))

        if isinstance(source, str) and source.startswith(("http://", "https://")):
            response = requests.get(source)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))

        raise ValueError("Unsupported input type for ImageLoader.to_pil")

    @staticmethod
    def to_tensor(source, size):
        t = transforms.ToTensor()
        return t(ImageLoader.to_pil(source).resize((size, size)))