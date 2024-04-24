def load_ascad(dataset_path):
    """
        Load the ASCAD dataset

        Parameters
        ----------
        dataset_path : str
            path to the dataset files

        Returns
        -------
        x_train : np.array
            training instances
        y_train : np.array
            training labels
        x_test : np.array
            testing instances
        x_test : np.array
            testing labels
    """

    try:
        x_train, y_train = load_mat('%s/train_32x32.mat' % dataset_path)
        x_test, y_test = load_mat('%s/test_32x32.mat' % dataset_path)
    except FileNotFoundError:
        print("Error: you need to download the SVHN files first.")
        sys.exit(-1)

    return x_train, y_train, x_test, y_test