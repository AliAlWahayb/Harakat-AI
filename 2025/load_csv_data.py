import os
import pandas as pd
import glob
from tqdm import tqdm

def load_csv_data_from_directory(directory_path, column_name='text_with_harakat'):
    """
    Load all CSV files from a directory and extract a specific column.
    
    Args:
        directory_path (str): Path to the directory containing CSV files
        column_name (str): Name of the column to extract (default: 'text_with_harakat')
    
    Returns:
        list: List of text samples extracted from all CSV files
    """
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {directory_path}")
    
    print(f"Found {len(csv_files)} CSV files in {directory_path}")
    
    # Initialize an empty list to store all text samples
    all_samples = []
    
    # Process each CSV file
    for csv_file in tqdm(csv_files, desc="Loading CSV files"):
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Check if the required column exists
            if column_name not in df.columns:
                print(f"Warning: Column '{column_name}' not found in {csv_file}. Skipping file.")
                continue
            
            # Extract the text samples and add to the list
            samples = df[column_name].dropna().tolist()
            all_samples.extend(samples)
            
        except Exception as e:
            print(f"Error processing file {csv_file}: {str(e)}")
    
    print(f"Successfully loaded {len(all_samples)} text samples from {len(csv_files)} CSV files")
    
    return all_samples

# Example usage:
if __name__ == "__main__":
    # Replace with your directory path
    directory_path = "data/"
    
    # Load all text samples
    samples = load_csv_data_from_directory(directory_path)
    
    # Print some statistics
    print(f"Total number of samples: {len(samples)}")
    if samples:
        print(f"Average sample length: {sum(len(s) for s in samples) / len(samples):.2f} characters")
        print(f"Sample example: {samples[0][:100]}...")