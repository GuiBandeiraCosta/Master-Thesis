import os
import shutil

# List of folder names
folder_names = [
    "3aa6fb7a",
    "36d67576",
    "e21d9049",
    "68b16354",
    "e8593010",
    "39e1d7f9",
    "c9f8e694",
    "22eb0ac0",
    "913fb3ed",
    "08ed6ac7"
]

# Get the current directory
current_directory = os.getcwd()

# Create folders and move files
for folder_name in folder_names:
    # Create the directory if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    
    # List all files in the current directory
    for filename in os.listdir(current_directory):
        # Check if the filename contains the folder name string
        
        if folder_name in filename and "json" in filename:
            # Move the file to the corresponding folder
            source = os.path.join(current_directory, filename)
            destination = os.path.join(current_directory, folder_name, filename)
            shutil.move(source, destination)
            print(f"Moved '{filename}' to '{folder_name}'")

print("All matching files have been moved.")
