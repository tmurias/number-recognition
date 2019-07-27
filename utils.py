import csv
import numpy

def csv_to_array(csv_filename):
    """Convert a csv file to a numpy array.
        csv_filename (str): Name of CSV file in current directory
    """
    val_array = []
    with open(csv_filename) as csv_file:
        reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            val_array.append(row)
    return numpy.array(val_array)

