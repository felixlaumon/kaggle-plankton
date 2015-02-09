from sklearn.cross_validation import StratifiedShuffleSplit


def straified_train_test_split(*arrays, **kwargs):
    """Like train_test_split but stratified
    Note that y is a required keyword argument
    """

    y = kwargs['y']
    test_size = kwargs.pop('test_size', 0.25)
    random_state = kwargs.pop('random_state', None)
    sss = StratifiedShuffleSplit(y, test_size=test_size, random_state=random_state)
    train, test = iter(sss).next()

    return flatten([[a[train], a[test]] for a in arrays])


def flatten(l):
    """http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python"""
    return [item for sublist in l for item in sublist]
