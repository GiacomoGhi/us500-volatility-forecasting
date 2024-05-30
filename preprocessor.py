import os
import csv
import math
from visual_util import ColoredPrint as cp

def ohlcDataPreprocessor(inputFilePath: str, outputDirPath: str) -> None:
    # Check if input file exists
    if not os.path.isfile(inputFilePath):
        cp.red(f"Error: Input file {inputFilePath} does not exist.")
        return
    
    # Ensure the output directory exists
    if not os.path.isdir(outputDirPath):
        os.makedirs(outputDirPath)
    
    # Check if output directory already has files
    if os.listdir(outputDirPath):
        cp.blue(f"Info: Output directory {outputDirPath} already contains files. Assuming processing is done.")
        return
    
    # Read the input file
    with open(inputFilePath, mode='r') as infile:
        reader = csv.reader(infile)
        headers = next(reader)
        rows = list(reader)
    
    # Prepare the headers and calculate volatility
    headers.append('Volatility')
    processed_data = []

    for row in rows:
        date, time, open_val, high, low, close = row
        open_val, high, low, close = map(float, [open_val, high, low, close])
        
        max_ohlc_val = max(open_val, high, low, close)
        min_ohlc_val = min(open_val, high, low, close)
        
        volatility = round(((max_ohlc_val - min_ohlc_val) / close) * 100, 2)
        
        # Format the time value without semicolons
        formatted_time = time.replace(':', '')
        new_row = [date, formatted_time, open_val, high, low, close, volatility]
        processed_data.append(new_row)
    
    # Determine split indices
    total_rows = len(processed_data)
    train_end = math.ceil(0.6 * total_rows)
    val_end = train_end + math.ceil(0.2 * total_rows)
    
    train_data = processed_data[:train_end]
    val_data = processed_data[train_end:val_end]
    test_data = processed_data[val_end:]

    # Define file paths
    train_file_path = os.path.join(outputDirPath, 'train.csv')
    val_file_path = os.path.join(outputDirPath, 'val.csv')
    test_file_path = os.path.join(outputDirPath, 'test.csv')

    # Write the data to the respective files
    for file_path, data in zip([train_file_path, val_file_path, test_file_path], [train_data, val_data, test_data]):
        with open(file_path, mode='w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(headers)
            writer.writerows(data)
    
    cp.green(f"Success: Processed data written to {outputDirPath} as train.csv, val.csv, and test.csv")

# Example usage
ohlcDataPreprocessor('data/D1-USA500IDXUSD.csv', 'processed-data/D1')
