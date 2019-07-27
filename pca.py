import utils
import numpy as np
from scipy.linalg import eigh

def main():
    x1600 = utils.csv_to_array('X1600.csv')
    paxes = [] # Array of principal axes for each label
    avgs = [] # Average of all the data points for each label

    for label in range(0, 10):
        start_col = 1600*label
        end_col = start_col + 1600
        data_points = x1600[:,start_col:end_col]
        pa, avg = train(data_points, 29)
        paxes.append(pa)
        avgs.append(avg)

    test_points = utils.csv_to_array('Te28.csv')
    test_labels = utils.csv_to_array('Lte28.csv').T

    num_points = np.size(test_points, 1)
    incorrect = 0 # Number of incorrectly classified points

    # Classify each test point and compare it to the real labels
    for col in range(num_points):
        x = test_points[:, col]

        errors = [] # The lowest error will correspond to the correct label

        # Evaluate how well this data point fits each label
        for label in range(10):

            # Tensor where columns are principal axis vectors for this label
            U = paxes[label]

            # Average of all training points that belong to this label
            mu = avgs[label]

            f = U.T @ (x - mu)

            # x_classified represents the data point x, conformed to a label
            # The closer x_classified resembles the original data point x,
            # the more likely that x belongs to this label
            x_classified = (U @ f) + mu

            # The smaller this error, the more likely this is our label
            e = np.linalg.norm(x - x_classified)
            errors.append(e)

        best_label = errors.index(min(errors))
        if best_label != test_labels[0,col]:
            incorrect += 1

    print('Misclassified: ' + str(incorrect) + ' out of ' + str(num_points))


def train(data_points, q):
    d = np.size(data_points, 0) # Number of rows (dimension of points)
    m = np.size(data_points, 1) # Number of columns (number of data points)

    mu = data_points.mean(1) # Column vector, average of the columns

    normalized = np.copy(data_points)
    for col in range(m):
        col_norm = data_points[:, col] - mu
        normalized[:, col] = col_norm
    covariance = (normalized @ normalized.T) / m

    # Eigenvectors corresponding to the q largest eigenvalues (ascending order)
    # These represent the principal axes
    _, Uq = eigh(covariance, eigvals=[d-q,d-1])

    # Switch to descending order
    Uq = np.fliplr(Uq)

    # Return the principal axes and the average of the data points
    return Uq, mu

if __name__ == '__main__':
    main()

