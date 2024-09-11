
import argparse
import torch

from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_video, export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


repo = "ByteDance/AnimateDiff-Lightning"
ckpt = f"animatediff_lightning_4step_diffusers.safetensors" # step options [1,2,4,8] -- don't forget to update below as well
base = "emilianJR/epiCRealism"

adapter = MotionAdapter().to("cuda", torch.float16)
adapter.load_state_dict(load_file(hf_hub_download(repo ,ckpt), device="cuda"))
pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")


def generate_video(prompt: str, output_path: str):
    output = pipe(prompt=prompt, guidance_scale=1.0, num_inference_steps=4)
    export_to_gif(output.frames[0], output_path, fps=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a prompt")
    parser.add_argument('--prompt', type=str, required=True, help='The prompt for video generation')
    parser.add_argument('--output_path', type=str, required=True, help='The path to save the generated video')

    args = parser.parse_args()

    generate_video(args.prompt, args.output_path)
