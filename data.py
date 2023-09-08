import os
import shutil

# Set the source directory containing the images
source_directory = 'jpg'

# Set the destination directory for the classified images
destination_directory = 'images'

# Create the destination directories if they don't exist
os.makedirs(destination_directory, exist_ok=True)

# Create the training and testing directories within the destination directory
training_directory = os.path.join(destination_directory, 'training')
os.makedirs(training_directory, exist_ok=True)
testing_directory = os.path.join(destination_directory, 'testing')
os.makedirs(testing_directory, exist_ok=True)

# Iterate over the image files in the source directory
for filename in os.listdir(source_directory):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Get the image index from the filename
        index = int(filename.split('_')[-1].split('.')[0]) - 1

        # Calculate the folder index based on the desired distribution
        folder_index = index // 80

        # Create the destination folder if it doesn't exist
        folder_path = os.path.join(training_directory, str(folder_index))
        os.makedirs(folder_path, exist_ok=True)

        # Copy the image file to the appropriate folder
        source_path = os.path.join(source_directory, filename)
        destination_path = os.path.join(folder_path, filename)
        shutil.copy2(source_path, destination_path)

# Move the first 40 images from each training folder to the testing folder
for folder_index in range(17):
    training_folder_path = os.path.join(training_directory, str(folder_index))
    testing_folder_path = os.path.join(testing_directory, str(folder_index))

    # Create the testing folder for the current class if it doesn't exist
    os.makedirs(testing_folder_path, exist_ok=True)

    # Get the list of image files in the training folder
    image_files = os.listdir(training_folder_path)

    # Move the first 40 images from the training folder to the testing folder
    for i in range(20):
        source_path = os.path.join(training_folder_path, image_files[i])
        destination_path = os.path.join(testing_folder_path, image_files[i])
        shutil.move(source_path, destination_path)

print('Classification completed successfully.')
