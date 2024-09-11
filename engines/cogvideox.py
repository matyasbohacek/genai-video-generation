
import argparse
import torch

from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video


pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()


def generate_video(prompt: str, output_path: str):
    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        guidance_scale=6,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]

    export_to_video(video, output_path, fps=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a prompt")
    parser.add_argument('--prompt', type=str, required=True, help='The prompt for video generation')
    parser.add_argument('--output_path', type=str, required=True, help='The path to save the generated video')

    args = parser.parse_args()

    generate_video(args.prompt, args.output_path)
