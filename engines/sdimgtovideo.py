import argparse
import os
import torch
from diffusers import StableDiffusion3Pipeline, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image


# Load Stable Diffusion for image generation
def load_sd_pipeline():
    sd_pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16
    )
    sd_pipe = sd_pipe.to("cuda")
    return sd_pipe


# Load Stable Video Diffusion for video generation
def load_vid_pipeline():
    vid_pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    vid_pipe.enable_model_cpu_offload()
    vid_pipe.unet.enable_forward_chunking()
    return vid_pipe


def generate_video(prompt: str, output_path: str):
    # Step 1: Generate the base image using Stable Diffusion (Text to Image)
    print(f"Generating base image for prompt: {prompt}")
    sd_pipe = load_sd_pipeline()
    base_image = sd_pipe(prompt).images[0]
    sd_pipe.to("cpu")  # Move the pipeline to CPU to free up GPU memory
    del sd_pipe  # Optionally delete the pipeline to free memory completely
    torch.cuda.empty_cache()  # Clear the cache

    # Optionally save the base image
    base_image.save("base_image.png")
    print("Base image saved as base_image.png")

    # Resize the image for the video generation (expected input size)
    base_image = base_image.resize((1024, 576))

    # Step 2: Generate the video using the image (Image to Video)
    print("Generating video based on the base image...")
    vid_pipe = load_vid_pipeline()
    generator = torch.manual_seed(42)  # For deterministic results
    frames = vid_pipe(base_image, decode_chunk_size=2, generator=generator, num_frames=25).frames[0]
    vid_pipe.to("cpu")  # Move the video pipeline to CPU to free up GPU memory
    del vid_pipe  # Optionally delete the pipeline to free memory completely
    torch.cuda.empty_cache()  # Clear the cache

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