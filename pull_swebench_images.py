#!/usr/bin/env python3
"""Pre-pull all Docker images for SWE-bench Lite."""

import subprocess
import sys
from datasets import load_dataset

def get_docker_image_name(instance_id: str) -> str:
    """Convert instance ID to Docker image name."""
    id_docker = instance_id.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{id_docker}:latest".lower()

def pull_image(image_name: str) -> bool:
    """Pull a single Docker image."""
    try:
        print(f"Pulling {image_name}...")
        result = subprocess.run(
            ["docker", "pull", image_name],
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout per image
        )
        if result.returncode == 0:
            print(f"✓ {image_name}")
            return True
        else:
            print(f"✗ Failed: {image_name}")
            print(f"  Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Exception pulling {image_name}: {e}")
        return False

def main():
    print("Loading SWE-bench Lite dataset...")
    dataset = load_dataset("princeton-nlp/SWE-Bench_Lite", split="test")
    
    print(f"Found {len(dataset)} instances")
    print("=" * 60)
    
    images = [get_docker_image_name(inst["instance_id"]) for inst in dataset]
    
    success_count = 0
    failed_images = []
    
    for i, image in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] ", end="")
        if pull_image(image):
            success_count += 1
        else:
            failed_images.append(image)
    
    print("\n" + "=" * 60)
    print(f"✓ Successfully pulled: {success_count}/{len(images)}")
    if failed_images:
        print(f"✗ Failed: {len(failed_images)}")
        print("\nFailed images:")
        for img in failed_images:
            print(f"  - {img}")
        sys.exit(1)
    else:
        print("All images pulled successfully!")

if __name__ == "__main__":
    main()
