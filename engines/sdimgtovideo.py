import argparse
import os

import torch
from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image


# Load the pipelines
def load_pipelines():
    # Load Stable Diffusion for image generation
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium",
        torch_dtype=torch.float16
    )
    sd_pipe = sd_pipe.to("cuda")

    # Load Stable Video Diffusion for video generation
    vid_pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        torch_dtype=torch.float16,
        variant="fp16"
    )

    # Optimize for GPU use
    vid_pipe.enable_model_cpu_offload()
    vid_pipe.unet.enable_forward_chunking()

    return sd_pipe, vid_pipe


def generate_video(prompt: str, output_path: str):
    # Load the models
    sd_pipe, vid_pipe = load_pipelines()

    # Step 1: Generate the base image using Stable Diffusion (Text to Image)
    print(f"Generating base image for prompt: {prompt}")
    base_image = sd_pipe(prompt).images[0]

    # Optionally save the base image
    base_image.save("base_image.png")
    print("Base image saved as base_image.png")

    # Resize the image for the video generation (expected input size)
    base_image = base_image.resize((1024, 576))

    # Step 2: Generate the video using the image (Image to Video)
    print("Generating video based on the base image...")
    generator = torch.manual_seed(42)  # For deterministic results
    frames = vid_pipe(base_image, decode_chunk_size=2, generator=generator, num_frames=25).frames[0]

    # Step 3: Save the generated video
    export_to_video(frames, output_path, fps=7)
    print(f"Video saved as {output_path}")
    os.remove("base_image.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a prompt")
    parser.add_argument('--prompt', type=str, required=True, help='The prompt for video generation')
    parser.add_argument('--output_path', type=str, required=True, help='The path to save the generated video')

    args = parser.parse_args()

    generate_video(args.prompt, args.output_path)