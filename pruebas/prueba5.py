import os

# Define the directories to search
directories = ['pede/train_images','pede/train_masks']

# Loop through each specified directory
for directory in directories:
    # Check if the directory exists
    if os.path.exists(directory):
        # Loop through all files in the directory
        for filename in os.listdir(directory):
            # Check if the filename starts with 'image_'
            if filename.startswith('image_'):
                # Construct the full file path
                file_path = os.path.join(directory, filename)
                # Remove the file
                os.remove(file_path)
                print(f'Removed {file_path}')
    else:
        print(f'{directory} does not exist.')

print("Finished removing images starting with 'image_'.")
