import os
import pandas

def file_to_dataframe(filename):
    csv_path = os.path.join(os.getcwd(), filename)
    return pandas.read_csv(csv_path)

