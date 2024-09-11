import argparse
import os

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2VidPipeline
from PIL import Image


# Load the pipes
def load_pipelines():
    # Load Stable Diffusion v3 for image generation
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3",
        torch_dtype=torch.float16
    )
    sd_pipe = sd_pipe.to("cuda")

    # Load Stable Video Diffusion for image-to-video generation
    vid_pipe = StableDiffusionImg2VidPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16
    )
    vid_pipe = vid_pipe.to("cuda")

    return sd_pipe, vid_pipe


def generate_video(prompt: str, output_path: str):
    # Load the models
    sd_pipe, vid_pipe = load_pipelines()

    # Step 1: Generate the base image using Stable Diffusion v3
    print(f"Generating base image for prompt: {prompt}")
    base_image = sd_pipe(prompt).images[0]

    # Optionally save the base image
    base_image.save("base_image.png")
    print("Base image saved as base_image.png")

    # Step 2: Generate the video using Stable Video Diffusion
    print("Generating video based on the base image...")
    video_frames = vid_pipe(prompt, input_image=base_image).frames

    # Step 3: Save the generated video as a GIF
    video_frames[0].save(
        output_path,
        save_all=True,
        append_images=video_frames[1:],
        duration=100,
        loop=0
    )
    print(f"Video saved as {output_path}")

    os.remove("base_image.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a prompt")
    parser.add_argument('--prompt', type=str, required=True, help='The prompt for video generation')
    parser.add_argument('--output_path', type=str, required=True, help='The path to save the generated video')

    args = parser.parse_args()

    generate_video(args.prompt, args.output_path)
