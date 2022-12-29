import torch


class MinimizerCallback:
    def __init__(
        self, mlp, loss_fn, writer,
        x_train, y_train,
        x_val=None, y_val=None
    ):
        self.writer = writer
        self.loss_fn = loss_fn
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.mlp = mlp
        self.iter = 0

    def __call__(self, parameter_state):
        with torch.no_grad():
            output = self.mlp(self.x_train)
            train_loss = self.loss_fn(output, self.y_train)

        self.writer.add_scalar(
            f'{self.loss_fn.__class__.__name__}/Train',
            train_loss, self.iter
        )

        if self.x_val is not None and self.y_val is not None:
            with torch.no_grad():
                output = self.mlp(self.x_val)
                val_loss = self.loss_fn(output, self.y_val)
            self.writer.add_scalar(
                f'{self.loss_fn.__class__.__name__}/Val',
                val_loss, self.iter
            )

        self.iter += 1