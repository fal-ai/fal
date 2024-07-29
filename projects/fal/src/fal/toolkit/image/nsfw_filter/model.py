import fal


@fal.cached
def get_model():
    import os

    from transformers import AutoModelForImageClassification, ViTImageProcessor

    os.environ["TRANSFORMERS_CACHE"] = "/data/models"
    os.environ["HF_HOME"] = "/data/models"

    model = AutoModelForImageClassification.from_pretrained(
        "Falconsai/nsfw_image_detection"
    )
    processor = ViTImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")

    return model, processor
