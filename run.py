import argparse
import os
import pandas as pd


# Dynamically import the generation functions based on the engine
def get_generation_function(engine):
    if engine == 'bytedance':
        from engines.bytedanceanimatediff import generate_video
    elif engine == 'cogvideox':
        from engines.cogvideox import generate_video
    elif engine == 'sdimgtovideo':
        from engines.sdimgtovideo import generate_video
    else:
        raise ValueError(f"Unsupported engine: {engine}")
    return generate_video


def run_batch_generation(prompt_csv, output_directory, engine, class_name_col, prompt_col, num_vids_per_class):
    # Load generation function based on the engine
    generate_video = get_generation_function(engine)

    # Check if output_directory exists
    if os.path.exists(output_directory):
        raise Exception(f"Output directory {output_directory} already exists. Aborting.")
    else:
        os.makedirs(output_directory)

    # Load the prompt CSV into a dataframe
    df = pd.read_csv(prompt_csv)

    # Iterate over the dataframe and generate videos for each class
    for index, row in df.iterrows():
        class_name = str(row[class_name_col])
        prompt = row[prompt_col]
        class_dir = os.path.join(output_directory, class_name)

        # Create a directory for the class if it doesn't exist
        os.makedirs(class_dir, exist_ok=True)

        # Generate num_vids_per_class videos for the class
        for i in range(num_vids_per_class):
            video_filename = chr(97 + i) + ".mp4"  # Names: a.mp4, b.mp4, c.mp4, etc.
            output_path = os.path.join(class_dir, video_filename)
            generate_video(prompt, output_path)
            print(f"Generated video {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process text-to-video generation")

    parser.add_argument('--prompt_csv', type=str, required=True, help='Path to the CSV file containing prompts')
    parser.add_argument('--output_directory', type=str, required=True, help='Directory to store generated videos')
    parser.add_argument('--engine', type=str, required=True, choices=['bytedance', 'cogvideox', 'sdimgtovideo'],
                        help='Engine to use for generation')
    parser.add_argument('--class_name_col', type=str, default="id", help='Column name for class names in the CSV')
    parser.add_argument('--prompt_col', type=str, default="name", help='Column name for prompts in the CSV')
    parser.add_argument('--num_vids_per_class', type=int, default=1, help='Number of videos to generate per class')

    args = parser.parse_args()

    run_batch_generation(
        prompt_csv=args.prompt_csv,
        output_directory=args.output_directory,
        engine=args.engine,
        class_name_col=args.class_name_col,
        prompt_col=args.prompt_col,
        num_vids_per_class=args.num_vids_per_class
    )


