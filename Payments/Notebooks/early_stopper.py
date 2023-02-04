# Early Stopper
import numpy as np


class EarlyStopper:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    https://github.com/Bjarten/early-stopping-pytorch
    """

    def __init__(self,
                 patience: int = 10,
                 verbose: bool = False,
                 delta: float = 0,
                 trace_func=print,
                 saved_model_name="model1"):
        """

        :param patience: How long to wait after last time validation loss improved.
                            Default: 10
        :param verbose: If True, prints a message for each validation loss improvement.
                            Default: False
        :param delta: Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        :param trace_func: trace print function.
                            Default: print
        :param saved_model_name: Name to where save this model, whitout the w2v extension
                            Default: model1
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
        self.saved_model_name = saved_model_name

    def __call__(self, loss, model):
        """Call for early stopper

        Args:
            loss (Number): Epoch loss
            model (Word2Vec Model): Model
        """

        if loss < 0:
            score = loss
        else:
            score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}'
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        """Saves model when validation loss decrease.

        Args:
            loss (Number): Epoch loss
            model (Word2Vec Model): Model
        """
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {loss:.6f}).  Saving model ...'
            )

        model.save(f"{self.saved_model_name}.w2v")
        self.val_loss_min = loss
