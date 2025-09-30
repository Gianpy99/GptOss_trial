class Dataset:
    def __init__(self, data_path, preprocess_fn=None):
        self.data_path = data_path
        self.preprocess_fn = preprocess_fn
        self.data = self.load_data()

    def load_data(self):
        # Load data from the specified path
        with open(self.data_path, 'r') as file:
            data = file.readlines()
        return data

    def preprocess(self):
        if self.preprocess_fn is not None:
            self.data = [self.preprocess_fn(line) for line in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]