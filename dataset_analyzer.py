import pandas as pd  # Importing pandas library for data manipulation and analysis
import matplotlib.pyplot as plt  # Importing matplotlib for data visualization
from pathlib import Path  # Importing Path from pathlib to handle file paths

class DatasetAnalyzer:
  def __init__(self, dataset_dir, file_name) -> None:
    """
    Initialize the DatasetAnalyzer with a directory and file name.
    
    Parameters:
    - dataset_dir: Directory where the dataset files are stored.
    - file_name: Name of the dataset file to be analyzed.
    """
    self.dataset_dir = Path(dataset_dir)  # Convert directory path to a Path object
    self.file_name = file_name  # Store the file name
    self.dataset_files = self._get_dataset_from_dir()  # Retrieve dataset files from the directory

  def _get_dataset_from_dir(self):
    """
    Retrieve dataset files from the specified directory.
    
    Returns:
    - List of dataset file paths that match the specified file name and are CSV files.
    """
    dataset_files = []  # Initialize an empty list to store dataset file paths
    if not self.dataset_dir.exists():  # Check if the directory exists
      print(f"Error: Directory '{self.dataset_dir}' does not exist.")
      return dataset_files  # Return empty list if directory does not exist

    # Iterate over all files in the directory matching the file_name
    for file_path in self.dataset_dir.rglob(self.file_name):
      if file_path.is_file() and file_path.suffix == '.csv':  # Check if the file is a CSV file
        dataset_files.append(str(file_path))  # Add file path to the list

    return dataset_files  # Return the list of dataset file paths

  def _read_and_validate_file(self, file_path, required_columns):
    """
    Read a CSV file and validate its columns.
    
    Parameters:
    - file_path: Path of the file to be read.
    - required_columns: List of columns that are required to be present in the file.
    
    Returns:
    - DataFrame if the file is valid and contains the required columns, otherwise None.
    """
    if not Path(file_path).exists():  # Check if the file exists
      print(f"Error: File '{file_path}' does not exist.")
      return None

    try:
      data = pd.read_csv(file_path)  # Try to read the CSV file
    except pd.errors.ParserError:
      print(f"Error: Failed to parse '{file_path}' as a CSV file.")
      return None

    # Check if all required columns are present in the file
    if not all(col in data.columns for col in required_columns):
      print(f"Error: File '{file_path}' does not contain required columns {required_columns}.")
      return None

    return data  # Return the DataFrame if valid

  def _plot_data(self, data, x_label, y_label, title, dist=False):
    """
    Plot the data using matplotlib.
    
    Parameters:
    - data: Data to be plotted.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - title: Title of the plot.
    - dist: Boolean to determine whether to plot a histogram (True) or a bar plot (False).
    """
    try:
      if dist:
        plt.hist(data, bins='auto')  # Plot a histogram if dist is True
      else:
        data.plot(kind='bar')  # Plot a bar plot if dist is False
      plt.xlabel(x_label)  # Set x-axis label
      plt.ylabel(y_label)  # Set y-axis label
      plt.title(title)  # Set plot title
      plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
      plt.show()  # Display the plot
    except Exception as e:
      print(f"Error during plotting: {e}")  # Print any errors that occur during plotting

  def plot_last_column_distribution(self):
    """
    Plot the distribution of the last column in each dataset file.
    """
    for file_path in self.dataset_files:  # Iterate over each dataset file
      data = self._read_and_validate_file(file_path, [])  # Read and validate the file
      if data is None:
        continue  # Skip to the next file if the current file is not valid

      last_column = data.iloc[:, -1]  # Get the last column of the DataFrame
      if not pd.api.types.is_numeric_dtype(last_column):  # Check if the last column is numeric
        try:
          last_column = pd.to_numeric(last_column)  # Convert to numeric if not already
        except ValueError:
          print(f"Error: Last column in '{file_path}' contains non-numeric data.")
          continue

      # Plot the distribution of the last column
      self._plot_data(last_column, last_column.name, "Frequency", f"Volatility values distribution in {file_path}", True)

  def plot_average_volatility_by_day_of_week(self):
    """
    Plot the average volatility by day of the week for each dataset file.
    """
    for file_path in self.dataset_files:  # Iterate over each dataset file
      data = self._read_and_validate_file(file_path, ['Date', 'Volatility'])  # Read and validate the file
      if data is None:
        continue  # Skip to the next file if the current file is not valid

      data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')  # Convert 'Date' column to datetime
      data['DayOfWeek'] = data['Date'].dt.day_name()  # Extract the day of the week from 'Date' column

      avg_volatility = data.groupby('DayOfWeek')['Volatility'].mean()  # Calculate the average volatility for each day of the week
      days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']  # Define the order of days
      avg_volatility = avg_volatility.reindex(days_order)  # Reindex to ensure correct order of days

      # Plot the average volatility by day of the week
      self._plot_data(avg_volatility, 'Day of the Week', 'Average Volatility', f"Average Volatility by Day of the Week in {file_path}")

  def plot_average_volatility_by_hour(self):
    """
    Plot the average volatility by hour of the day for each dataset file.
    """
    for file_path in self.dataset_files:  # Iterate over each dataset file
      data = self._read_and_validate_file(file_path, ['Time', 'Volatility'])  # Read and validate the file
      if data is None:
        continue  # Skip to the next file if the current file is not valid

      # Filter out invalid 'Time' entries and pad with zeros to ensure proper formatting
      data = data[data['Time'].apply(lambda x: str(x).zfill(6).isdigit())]
      data['Time'] = pd.to_datetime(data['Time'].apply(lambda x: str(x).zfill(6)), format='%H%M%S').dt.hour  # Convert 'Time' to hour

      avg_volatility = data.groupby('Time')['Volatility'].mean()  # Calculate the average volatility for each hour

      # Plot the average volatility by hour of the day
      self._plot_data(avg_volatility, 'Hour of the Day', 'Average Volatility', f"Average Volatility by Hour of the Day in {file_path}")

# Example usage
if __name__ == '__main__':
  analyzer = DatasetAnalyzer("./processed-data/", "train.csv")  # Create an instance of DatasetAnalyzer
  
  analyzer.plot_last_column_distribution()  # Plot distribution of the last column
  analyzer.plot_average_volatility_by_day_of_week()  # Plot average volatility by day of the week
  analyzer.plot_average_volatility_by_hour()  # Plot average volatility by hour of the day
