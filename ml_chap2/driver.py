from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import transform_utils

def main():
    full_data_frame = transform_utils.file_to_dataframe("housing.csv")

    print(full_data_frame["ocean_proximity"].value_counts())
    print(full_data_frame.describe())

    full_data_frame.hist(bins=50, figsize=(20,15))
    plt.show()

    scatter_matrix(full_data_frame[["median_house_value", "median_income"]], figsize=(12, 8))
    plt.show()

if __name__ == '__main__':
    main()

