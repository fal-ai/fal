import fal
from fal.toolkit import Image

def get_random_image(size: tuple = (1024, 1024)):
    import numpy as np
    from PIL import Image
    import base64
    from io import BytesIO

    data = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(data)
    return img

class T(fal.App):
    requirements = ["pillow", "numpy"]

    @fal.endpoint("/")
    def endpoint(self) -> list[Image]:
        images = [get_random_image() for _ in range(3)]
        fal_images = [Image.from_pil(ip , format="jpeg") for ip in images]
        return fal_images
