import os

def mkdir(*folder_sequence):
    """
    Recursively creates directories based on a sequence of folder names.

    Args:
        folder_sequence (list or tuple): A sequence of folder names to create hierarchically.
    
    Returns:
        output_folder_path (str): The full path of the created directory.
    """
    # Join the base folder with the sequence of folders
    output_folder_path = os.path.join(*folder_sequence)

    # Recursively create the directories, ignoring if they already exist
    os.makedirs(output_folder_path, exist_ok=True)
    
    return output_folder_path
