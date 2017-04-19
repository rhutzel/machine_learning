import matplotlib.pyplot
import tkinter
import transform_utils

def main():
    full_data_frame = transform_utils.file_to_dataframe("housing.csv")
    print(full_data_frame["ocean_proximity"].value_counts())
    print(full_data_frame.describe())
    full_data_frame.hist(bins=50, figsize=(20,15))

if __name__ == '__main__':
    main()

