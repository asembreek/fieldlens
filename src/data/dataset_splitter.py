class DatasetSplitter:
    def __init__(
        self, scaler, train_mask, test_mask, passthrough_cols=None, drop_cols=None
    ):
        self.scaler = scaler
        self.passthrough_cols = passthrough_cols
        self.drop_cols = drop_cols

    def fit(self, X):
        pass

    def train_test_split_transform(self, X):
        pass

    def fit_train_test_split_transform(self, X):
        pass
