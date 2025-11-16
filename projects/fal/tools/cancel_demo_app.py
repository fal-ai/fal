"""
Fal App demonstrating distributed worker cancellation with SDXL.

This app shows how to:
1. Cancel SDXL generation mid-inference using is_cancelled()
2. Keep workers alive after cancellation
3. Accept new work after cancellation

Usage:
    fal run cancel_demo_app.py

Then use the /cancel endpoint to cancel running generation.
"""
from math import floor, sqrt
from typing import TYPE_CHECKING, Any

import fal
from fal.toolkit import File, Image
from pydantic import BaseModel, Field

from fal.distributed import DistributedRunner, DistributedWorker

if TYPE_CHECKING:
    import torch
    from PIL import Image as PILImage


def tensors_to_image_grid(
    tensors: list["torch.Tensor"], blur_radius: int = 0
) -> "PILImage.Image":
    """
    Convert a list of tensors to a grid image.
    """
    import torchvision  # type: ignore[import-untyped]
    from PIL import Image as PILImage
    from PIL import ImageFilter

    # Create a grid of images
    image = (
        torchvision.utils.make_grid(
            tensors,
            nrow=floor(sqrt(len(tensors))),
            normalize=True,
            scale_each=True,
        )
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    image = (image * 255).astype("uint8")
    pil_image = PILImage.fromarray(image)

    if blur_radius > 0:
        pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return pil_image


class CancellableSDXLWorker(DistributedWorker):
    """
    Worker that runs SDXL with cancellation support.
    Each worker uses a different seed to generate parallel images.
    """

    def setup(self, **kwargs: Any) -> None:
        """Initialize SDXL pipeline."""
        import torch
        from diffusers import AutoencoderTiny, StableDiffusionXLPipeline

        self.rank_print("Loading SDXL pipeline...")
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            low_cpu_mem_usage=False,
        ).to(self.device, dtype=torch.float16)
        
        self.tiny_vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesdxl",
            torch_dtype=torch.float16,
        ).to(self.device, dtype=torch.float16)
        
        if self.rank != 0:
            self.pipeline.set_progress_bar_config(disable=True)
        
        self.rank_print("✅ SDXL pipeline loaded")

    def pipeline_callback(
        self,
        _pipeline: "torch.nn.Module",
        step: int,
        timestep: int,
        tensors: dict[str, "torch.Tensor"],
    ) -> dict[str, "torch.Tensor"]:
        """
        Callback during generation - CHECK FOR CANCELLATION HERE!
        """
        import torch
        import torch.distributed as dist

        # ⚠️ KEY FEATURE: Check for cancellation during generation
        if self.is_cancelled():
            self.rank_print(f"⚠️ CANCELLED at step {step}")
            # Raise an exception to stop the pipeline
            raise InterruptedError("Generation cancelled by user")

        # Stream intermediate results every 5 steps
        if step > 0 and step % 5 != 0:
            return tensors

        latents = tensors["latents"]
        image = self.tiny_vae.decode(
            latents / self.tiny_vae.config.scaling_factor, return_dict=False
        )[0]
        image = self.pipeline.image_processor.postprocess(image, output_type="pt")[0]

        if self.rank == 0:
            gather_list = [
                torch.zeros_like(image, device=self.device)
                for _ in range(self.world_size)
            ]
        else:
            gather_list = None

        dist.gather(image, gather_list, dst=0)

        if gather_list:
            remaining = timestep / 1000
            grid_image = tensors_to_image_grid(
                gather_list, blur_radius=int(remaining * 10)
            )
            self.add_streaming_result({"image": grid_image, "step": step})

        dist.barrier()
        return tensors

    def __call__(
        self,
        streaming: bool = False,
        width: int = 1024,
        height: int = 1024,
        prompt: str = "A fantasy landscape",
        negative_prompt: str = "blurry, low quality",
        num_inference_steps: int = 50,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run SDXL generation with cancellation support.
        Each worker uses rank as seed for different images.
        """
        import torch
        import torch.distributed as dist

        self.rank_print(f"Starting generation: {prompt}")
        
        try:
            image = self.pipeline(
                prompt=prompt,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                output_type="pt",
                generator=torch.Generator(device=self.device).manual_seed(self.rank),
                callback_on_step_end=self.pipeline_callback if streaming else None,
            ).images[0]

            if self.rank == 0:
                gather_list = [
                    torch.zeros_like(image, device=self.device)
                    for _ in range(self.world_size)
                ]
            else:
                gather_list = None

            # Gather images from all workers
            dist.gather(image, gather_list, dst=0)

            # Clean memory
            torch.cuda.empty_cache()

            if not gather_list:
                return {}

            # Main worker returns the grid
            grid_image = tensors_to_image_grid(gather_list)
            self.rank_print("✅ Generation completed")
            return {"status": "completed", "image": grid_image}

        except InterruptedError:
            # Generation was cancelled
            self.rank_print("⚠️ Generation was cancelled")
            torch.cuda.empty_cache()
            return {"status": "cancelled", "message": "Generation cancelled by user"}


class GenerationRequest(BaseModel):
    """Request for SDXL generation."""

    prompt: str = Field(default="A fantasy landscape with mountains and a castle")
    negative_prompt: str = Field(default="blurry, low quality")
    num_inference_steps: int = Field(default=50)
    width: int = Field(default=1024)
    height: int = Field(default=1024)


class GenerationResponse(BaseModel):
    """Response from generation."""

    status: str = Field(description="completed or cancelled")
    image: File | None = Field(default=None)
    message: str | None = Field(default=None)


class CancellableSDXLApp(fal.App):
    machine_type = "GPU-H100"
    num_gpus = 2
    requirements = [
        "accelerate",
        "diffusers",
        "fal",
        "torch==2.6.0+cu124",
        "torchvision==0.21.0+cu124",
        "transformers",
        "pyzmq",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu124",
    ]

    async def setup(self) -> None:
        """Initialize distributed runner with SDXL workers."""
        self.runner = DistributedRunner(
            worker_cls=CancellableSDXLWorker,
            world_size=self.num_gpus,
        )
        await self.runner.start()

        # Warmup with quick generation
        print("🔥 Warming up with quick generation...")
        warmup = await self.runner.invoke(
            {
                "prompt": "a cat",
                "num_inference_steps": 1,
                "width": 512,
                "height": 512,
            }
        )
        assert "status" in warmup, "Warmup failed"
        print("✅ Warmup complete!")

    @fal.endpoint("/")
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate images using SDXL across multiple GPUs.
        Each GPU generates with a different seed, results are combined in a grid.
        
        Generation can be cancelled using the /cancel endpoint.
        """
        result = await self.runner.invoke(request.dict())
        
        if result.get("status") == "cancelled":
            return GenerationResponse(
                status="cancelled",
                message=result.get("message", "Generation was cancelled")
            )
        
        return GenerationResponse(
            status="completed",
            image=Image.from_pil(result["image"]),
        )

    @fal.endpoint("/cancel")
    async def cancel(self) -> dict[str, str]:
        """
        Cancel any currently running generation.

        This demonstrates the cancellation feature!
        - Workers check is_cancelled() during each diffusion step
        - Generation stops immediately at the next step
        - Workers stay alive and ready for new generations
        """
        await self.runner.cancel()
        return {
            "status": "cancelled",
            "message": "Cancellation signal sent to all workers. Generation will stop at next step.",
        }


if __name__ == "__main__":
    app = fal.wrap_app(CancellableSDXLApp)
    app()
