#!/usr/bin/env python3
"""Test script to verify the pure Python REPL environment works correctly.

Run with: python test_pure_python_repl.py

This will:
1. Start a simple Docker container
2. Execute pure Python code inside it
3. Verify pathlib, subprocess work as expected
"""

import subprocess
import tempfile
import os
import sys

def test_repl_in_docker():
    """Test that Python code executes correctly inside Docker."""
    
    # Use a simple Python image for testing
    image = "python:3.11-slim"
    
    print("=" * 60)
    print("Testing Pure Python REPL Environment")
    print("=" * 60)
    
    # Pull the image first
    print("\n1. Pulling Docker image...")
    result = subprocess.run(
        ["docker", "pull", image],
        capture_output=True,
        text=True,
        timeout=120
    )
    if result.returncode != 0:
        print(f"Failed to pull image: {result.stderr}")
        return False
    print("   Done!")
    
    # Start a container
    print("\n2. Starting Docker container...")
    result = subprocess.run(
        ["docker", "run", "-d", "--rm", image, "sleep", "300"],
        capture_output=True,
        text=True,
        timeout=30
    )
    if result.returncode != 0:
        print(f"Failed to start container: {result.stderr}")
        return False
    
    container_id = result.stdout.strip()
    print(f"   Container ID: {container_id[:12]}")
    
    try:
        # Test 1: Basic Python execution
        print("\n3. Testing basic Python execution...")
        test_code = '''
import sys
print(f"Python version: {sys.version}")
print("Hello from Docker!")
'''
        result = subprocess.run(
            ["docker", "exec", container_id, "python", "-c", test_code],
            capture_output=True,
            text=True,
            timeout=30
        )
        print(f"   Output: {result.stdout.strip()}")
        if "Hello from Docker!" not in result.stdout:
            print("   FAILED: Basic execution failed")
            return False
        print("   PASSED!")
        
        # Test 2: pathlib file operations
        print("\n4. Testing pathlib file operations...")
        test_code = '''
from pathlib import Path

# Create a test file
test_file = Path("/tmp/test_file.txt")
test_file.write_text("Hello from pathlib!")

# Read it back
content = test_file.read_text()
print(f"Content: {content}")

# Check exists
print(f"Exists: {test_file.exists()}")

# List files
files = list(Path("/tmp").glob("*.txt"))
print(f"Files: {[f.name for f in files]}")
'''
        result = subprocess.run(
            ["docker", "exec", container_id, "python", "-c", test_code],
            capture_output=True,
            text=True,
            timeout=30
        )
        print(f"   Output:\n{result.stdout}")
        if "Hello from pathlib!" not in result.stdout:
            print("   FAILED: pathlib operations failed")
            return False
        print("   PASSED!")
        
        # Test 3: subprocess execution
        print("\n5. Testing subprocess execution...")
        test_code = '''
import subprocess

result = subprocess.run(
    ["echo", "Hello from subprocess"],
    capture_output=True,
    text=True
)
print(f"subprocess output: {result.stdout.strip()}")
print(f"return code: {result.returncode}")
'''
        result = subprocess.run(
            ["docker", "exec", container_id, "python", "-c", test_code],
            capture_output=True,
            text=True,
            timeout=30
        )
        print(f"   Output:\n{result.stdout}")
        if "Hello from subprocess" not in result.stdout:
            print("   FAILED: subprocess failed")
            return False
        print("   PASSED!")
        
        # Test 4: State persistence via pickle
        print("\n6. Testing state persistence via pickle...")
        # First execution - create variables
        test_code_1 = '''
import pickle

# Create some variables
my_var = "test_value"
my_list = [1, 2, 3]
my_dict = {"key": "value"}

# Save state
state = {"my_var": my_var, "my_list": my_list, "my_dict": my_dict}
with open("/tmp/state.pkl", "wb") as f:
    pickle.dump(state, f)
print("State saved!")
'''
        result = subprocess.run(
            ["docker", "exec", container_id, "python", "-c", test_code_1],
            capture_output=True,
            text=True,
            timeout=30
        )
        print(f"   Save output: {result.stdout.strip()}")
        
        # Second execution - load variables
        test_code_2 = '''
import pickle

# Load state
with open("/tmp/state.pkl", "rb") as f:
    state = pickle.load(f)

# Restore variables
my_var = state["my_var"]
my_list = state["my_list"]
my_dict = state["my_dict"]

print(f"my_var: {my_var}")
print(f"my_list: {my_list}")
print(f"my_dict: {my_dict}")
'''
        result = subprocess.run(
            ["docker", "exec", container_id, "python", "-c", test_code_2],
            capture_output=True,
            text=True,
            timeout=30
        )
        print(f"   Load output:\n{result.stdout}")
        if "test_value" not in result.stdout:
            print("   FAILED: State persistence failed")
            return False
        print("   PASSED!")
        
        # Test 5: Context simulation
        print("\n7. Testing context variable simulation...")
        test_code = '''
import json

# Simulate context loading (like the REPL env does)
context_data = {"issue": "Fix the bug in module.py", "repo_path": "/testbed"}
context_json = json.dumps(context_data)

# Write to file
with open("/tmp/context.json", "w") as f:
    f.write(context_json)

# Load it back (simulating what the REPL does)
with open("/tmp/context.json", "r") as f:
    context = json.load(f)

print(f"Issue: {context['issue']}")
print(f"Repo path: {context['repo_path']}")
'''
        result = subprocess.run(
            ["docker", "exec", container_id, "python", "-c", test_code],
            capture_output=True,
            text=True,
            timeout=30
        )
        print(f"   Output:\n{result.stdout}")
        if "Fix the bug" not in result.stdout:
            print("   FAILED: Context simulation failed")
            return False
        print("   PASSED!")
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe pure Python REPL environment is working correctly.")
        print("You can now run the mini-swe-agent with the new configuration.")
        return True
        
    finally:
        # Cleanup
        print("\nCleaning up...")
        subprocess.run(["docker", "stop", container_id], capture_output=True, timeout=60)
        print("Done!")


if __name__ == "__main__":
    try:
        success = test_repl_in_docker()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

