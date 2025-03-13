class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum improvement to be considered an actual improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if there is an improvement
        else:
            self.counter += 1  # Increment counter if no improvement

        if self.counter >= self.patience:
            print("Early stopping triggered.")
            self.early_stop = True
