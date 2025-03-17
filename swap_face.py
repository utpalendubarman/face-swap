import subprocess
# swap_face(source="face2.jpg",target="face1.jpg",output="res.jpg")
# Function to call the original command
def swap_face(source,target,output):
    # Define the command as if it's typed in the terminal
    command = [
        "python3",  # Or "python3" depending on your environment
        "run.py",  # Replace with the path to your script
        "-s", f"{source}",  # Source image
        "-t", f"{target}",  # Target image or video
        "-o", f"{output}",  # Output directory
        "--execution-provider", "cpu",  # Execution provider
        "--execution-threads", "4",  # Number of threads
    ]
    try:
        # Use subprocess to run the command
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Print the command output
        print("Command output:", result.stdout.decode())
        
        # Print any errors
        print("Command errors:", result.stderr.decode())
        
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running the command: {e}")
        print(f"Output: {e.output.decode()}")
        print(f"Error: {e.stderr.decode()}")
    

