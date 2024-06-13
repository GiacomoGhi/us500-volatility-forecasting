import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_last_column_distribution(file_path):
    """
    This function checks if a file exists and is a CSV, then plots the distribution of the last column.

    Args:
        file_path (str): The path to the CSV file.
    """
    # Check if file exists
    if not Path(file_path).exists():
        print(f"Error: File '{file_path}' does not exist.")
        return

    # Check if file extension is CSV
    if not file_path.endswith(".csv"):
        print(f"Error: File '{file_path}' is not a CSV file.")
        return

    # Try to read the CSV file using pandas
    try:
        data = pd.read_csv(file_path)
    except pd.errors.ParserError:
        print(f"Error: Failed to parse '{file_path}' as a CSV file.")
        return

    # Check if there are any columns
    if len(data.columns) == 0:
        print(f"Error: File '{file_path}' has no columns.")
        return

    # Extract the last column
    last_column = data.iloc[:, -1]

    # Check if the last column is numeric
    if not pd.api.types.is_numeric_dtype(last_column):
        try:
            last_column = pd.to_numeric(last_column)
        except ValueError:
            print(f"Error: Last column in '{file_path}' contains non-numeric data.")
            return

    # Plot the histogram of the last column using Matplotlib directly
    try:
        plt.hist(last_column, bins='auto')
        plt.xlabel(last_column.name)
        plt.ylabel("Frequency")
        plt.title(f"Volatility values distribution")
        plt.show()
    except Exception as e:
        print(f"Error during plotting: {e}")

# Example usage
if __name__ == '__main__':
    file_path = "./processed-data/D1/train.csv"  # Replace with your actual file path
    plot_last_column_distribution(file_path)
