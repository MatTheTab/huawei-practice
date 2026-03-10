import mindspore as ms
from mindspore.dataset import FashionMnistDataset
from mindspore.dataset.vision.transforms import Resize, HWC2CHW
from mindspore.dataset.transforms import TypeCast, OneHot
from mindspore.dataset.vision.utils import Inter
import numpy as np
import mindspore.nn as nn
from mindspore import ops
import matplotlib.pyplot as plt
from mindspore.train.callback import LossMonitor
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore import nn
from mindspore.train import Model
from mindspore.nn import Accuracy, Recall, F1, Precision


def create_dataset(
    data_dir, usage, num_parallel_workers, shuffle, height, width, batch_size=32
):
    resize_op = Resize((height, width), interpolation=Inter.BICUBIC)
    channel_flip_op = HWC2CHW()
    cast_type_op_int = TypeCast(np.int16)
    cast_type_op_float = TypeCast(np.float32)
    one_hot_op = OneHot(num_classes=10)

    fashion_mnist = FashionMnistDataset(
        data_dir,
        usage=usage,
        num_parallel_workers=num_parallel_workers,
        shuffle=shuffle,
    )
    fashion_mnist = fashion_mnist.map(
        operations=[resize_op, channel_flip_op, cast_type_op_float],
        input_columns="image",
        num_parallel_workers=8,
    )
    fashion_mnist = fashion_mnist.map(
        operations=[cast_type_op_int, one_hot_op, cast_type_op_float],
        input_columns="label",
        num_parallel_workers=8,
    )
    fashion_mnist = fashion_mnist.batch(
        batch_size=batch_size, drop_remainder=True, num_parallel_workers=8
    )
    return fashion_mnist


class CNN(nn.Cell):
    def __init__(self, input_shape, height, width, features, lin_features, num_classes):
        super(CNN, self).__init__()
        self.input_shape = input_shape
        self.height = height
        self.width = width
        self.relu = nn.LeakyReLU()
        self.cnn = []
        self.linear = []

        prev_channels = features[0]
        for idx in range(1, len(features)):
            curr_channels = features[idx]
            self.cnn.append(
                nn.Conv2d(
                    in_channels=prev_channels,
                    out_channels=curr_channels,
                    kernel_size=(3, 3),
                    pad_mode="valid",
                    stride=2,
                )
            )
            prev_channels = curr_channels
        self.cnn = nn.SequentialCell(self.cnn)
        flat_dims = self._get_flat_out()
        self.linear.append(nn.Linear(flat_dims, lin_features[0]))
        prev_dims = lin_features[0]
        for idx in range(1, len(lin_features)):
            self.linear.append(nn.Linear(prev_dims, lin_features[idx]))
            prev_dims = lin_features[idx]
        self.linear.append(nn.Linear(prev_dims, num_classes))
        self.linear = nn.SequentialCell(self.linear)

    def _get_flat_out(self):
        dummy_in = ops.expand_dims(
            ops.uniform(self.input_shape, minval=ms.Tensor(0.0), maxval=ms.Tensor(1.0)),
            axis=0,
        )
        dummy_out = self.cnn(dummy_in)
        flat_out = ops.flatten(dummy_out, start_dim=1)
        return flat_out.shape[-1]

    def construct(self, x):
        for conv_layer in self.cnn:
            x = conv_layer(x)
            x = self.relu(x)
        x = ops.flatten(x, start_dim=1)
        for lin_layer in self.linear[:-1]:
            x = lin_layer(x)
            x = self.relu(x)
        last_layer = self.linear[-1]
        x = last_layer(x)
        return x


if __name__ == "__main__":
    ms.set_device("CPU")

    train_fashion_mnist = create_dataset(
        "./data/fashion",
        usage="train",
        num_parallel_workers=8,
        shuffle=True,
        height=64,
        width=64,
    )
    test_fashion_mnist = create_dataset(
        "./data/fashion",
        usage="test",
        num_parallel_workers=8,
        shuffle=False,
        height=64,
        width=64,
    )

    cnn = CNN(
        input_shape=(1, 64, 64),
        height=64,
        width=64,
        features=[1, 8, 32, 64],
        lin_features=[128, 64, 32],
        num_classes=10,
    )

    print(cnn)

    metrics = {
        "Accuracy": Accuracy(),
        "Recall": Recall("classification"),
        "Precision": Precision("classification"),
        "F1": F1(),
    }

    loss = nn.SoftmaxCrossEntropyWithLogits()
    optim = nn.Momentum(
        params=cnn.trainable_params(), learning_rate=0.00002, momentum=0.99
    )
    model = Model(cnn, loss_fn=loss, optimizer=optim, metrics=metrics)
    config_ck = CheckpointConfig(save_checkpoint_steps=2000, keep_checkpoint_max=10)
    ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)
    model.train(
        10,
        train_fashion_mnist,
        callbacks=[ckpoint, LossMonitor(1000)],
        dataset_sink_mode=False,
    )

    print("Evaluating...")
    results = model.eval(test_fashion_mnist, dataset_sink_mode=False)
    print(f"Metrics: {results}")

    data_iter = test_fashion_mnist.create_tuple_iterator()
    images, labels = next(data_iter)
    output = model.predict(images)
    predicted_classes = ops.argmax(output, dim=1)

    print(f"Predicted: {predicted_classes[0].asnumpy()}")
    print(f"Actual:    {labels[0].asnumpy()}")
    plt.imshow(ops.squeeze(images[0]).asnumpy(), cmap="grey")
    plt.show()
