import argparse
import importlib.util
from pathlib import Path


def _load_pipeline_function(script_name, function_name):
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, function_name):
        raise AttributeError(f"{function_name} not found in {script_name}")

    return getattr(module, function_name)


def run_audio_pipeline():
    pipeline_fn = _load_pipeline_function('ForensiCore-audio-train.py', 'run_audio_pipeline')
    pipeline_fn()


def run_video_pipeline():
    pipeline_fn = _load_pipeline_function('ForensiCore-video-train.py', 'run_video_pipeline')
    pipeline_fn()


def main():
    parser = argparse.ArgumentParser(description='ForensiCore training launcher')
    parser.add_argument(
        '--pipeline',
        choices=['audio', 'video', 'both'],
        default='both',
        help='Choose which training pipeline to run',
    )
    args = parser.parse_args()

    if args.pipeline == 'audio':
        run_audio_pipeline()
    elif args.pipeline == 'video':
        run_video_pipeline()
    else:
        run_audio_pipeline()
        run_video_pipeline()


if __name__ == '__main__':
    main()
