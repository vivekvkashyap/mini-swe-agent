#!/usr/bin/env python3
"""Pre-pull all Docker images for SWE-bench Lite."""

import subprocess
import sys
from datasets import load_dataset

def get_docker_image_name(instance_id: str) -> str:
    """Convert instance ID to Docker image name."""
    id_docker = instance_id.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{id_docker}:latest".lower()

def image_exists_locally(image_name: str) -> bool:
    """Check if image exists locally without querying Docker Hub."""
    try:
        # Docker stores images without the "docker.io/" prefix
        # So we need to check both the full name and the short name
        result = subprocess.run(
            ["docker", "images", "-q", image_name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.stdout.strip():
            return True
        
        # Try without docker.io/ prefix
        short_name = image_name.replace("docker.io/", "")
        result = subprocess.run(
            ["docker", "images", "-q", short_name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False

def pull_image(image_name: str) -> tuple[bool, bool]:
    """Pull a single Docker image.
    
    Returns:
        tuple[bool, bool]: (success, was_skipped)
    """
    # Check if image already exists locally (doesn't count against rate limit!)
    if image_exists_locally(image_name):
        print(f"⊘ Already exists locally, skipping")
        return (True, True)
    
    # Image doesn't exist, need to pull from Docker Hub
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
            return (True, False)
        else:
            print(f"✗ Failed: {image_name}")
            print(f"  Error: {result.stderr}")
            return (False, False)
    except Exception as e:
        print(f"✗ Exception pulling {image_name}: {e}")
        return (False, False)

def main():
    print("Loading SWE-bench Lite dataset...")
    dataset = load_dataset("princeton-nlp/SWE-Bench_Lite", split="test")
    
    print(f"Found {len(dataset)} instances")
    print("=" * 60)
    
    images = [get_docker_image_name(inst["instance_id"]) for inst in dataset]
    
    success_count = 0
    skipped_count = 0
    failed_images = []
    
    for i, image in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] ", end="")
        success, was_skipped = pull_image(image)
        if success:
            success_count += 1
            if was_skipped:
                skipped_count += 1
        else:
            failed_images.append(image)
    
    print("\n" + "=" * 60)
    print(f"✓ Successfully pulled: {success_count}/{len(images)}")
    print(f"⊘ Skipped (already local): {skipped_count}/{len(images)}")
    print(f"↓ Downloaded from Docker Hub: {success_count - skipped_count}/{len(images)}")
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
