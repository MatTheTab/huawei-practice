import mindspore as ms
from mindspore.dataset.transforms import TypeCast
from mindspore.dataset.vision.transforms import HWC2CHW, Resize
from mindspore.dataset import FashionMnistDataset
from mindspore.dataset.vision.utils import Inter
import numpy as np
import mindspore.nn as nn
import math
from mindspore import ops
from mindspore.ops import binary_cross_entropy_with_logits
import matplotlib.pyplot as plt
from mindspore.dataset.vision import Rescale


def get_dataset(dataset_dir):
    fashion_ds = FashionMnistDataset(
        dataset_dir, usage="all", num_parallel_workers=8, shuffle=True
    )
    rescale_op = Rescale(1.0 / 255.0, 0.0)
    typecast_op = TypeCast(np.float32)
    resize_op = Resize((64, 64), interpolation=Inter.BICUBIC)
    channel_flip_op = HWC2CHW()
    fashion_ds = fashion_ds.map(
        operations=[typecast_op, resize_op, rescale_op, channel_flip_op],
        input_columns="image",
    )
    fashion_ds = fashion_ds.batch(32)
    return fashion_ds


class Generator(nn.Cell):
    def __init__(self, input_size, features, output_shape):
        super(Generator, self).__init__()
        self.input_shape = int(input_size)
        prev_shape = input_size
        layers = []
        for feature in features:
            layers.append(nn.Linear(prev_shape, feature))
            layers.append(nn.LeakyReLU())
            prev_shape = feature
        layers.append(nn.Linear(prev_shape, math.prod(output_shape)))
        self.layers = nn.SequentialCell(layers)
        self.output_shape = output_shape

    def construct(self, batch_size):
        x = ops.normal(
            (
                batch_size,
                self.input_shape,
            ),
            mean=0,
            stddev=0.5,
        )
        for layer in self.layers:
            x = layer(x)
        x = x.reshape((batch_size, *self.output_shape))
        return ops.sigmoid(x)


class Discriminator(nn.Cell):
    def __init__(self, input_shape, features):
        super(Discriminator, self).__init__()
        layer = []
        self.input_shape = input_shape
        input_channels = input_shape[0]
        prev_feature = input_channels
        for feature in features:
            layer.append(
                nn.Conv2d(
                    prev_feature,
                    feature,
                    kernel_size=(3, 3),
                    stride=2,
                    pad_mode="valid",
                )
            )
            prev_feature = feature
        self.layer = nn.SequentialCell(layer)
        prev_dim = self._get_flattened()
        self.linears = nn.SequentialCell([nn.Linear(prev_dim, 32), nn.Linear(32, 1)])

    def construct(self, x):
        x = self.layer(x)
        x = ops.flatten(x)
        x = self.linears(x)
        return x

    def _get_flattened(self):
        dummy_input = ops.expand_dims(
            ops.uniform(self.input_shape, minval=ms.tensor(0.0), maxval=ms.tensor(1.0)),
            axis=0,
        )
        dummy_out = self.layer(dummy_input)
        return ops.flatten(dummy_out).shape[-1]


if __name__ == "__main__":
    fashion_ds = get_dataset("./data/fashion")
    ds = Discriminator(input_shape=(1, 64, 64), features=[8, 16, 32])
    gen = Generator(input_size=64, features=[128, 256, 512], output_shape=(1, 64, 64))

    print(ds)
    print(gen)

    optimizer_ds = nn.AdamWeightDecay(ds.trainable_params())
    optimizer_gen = nn.AdamWeightDecay(gen.trainable_params())

    def forward_pass_ds(real):
        batch_size = real.shape[0]
        out_real = ds(real)
        label_real = ops.ones_like(out_real)
        loss_real = ops.binary_cross_entropy_with_logits(out_real, label_real)
        fake = gen(batch_size)
        out_fake = ds(fake)
        label_fake = ops.zeros_like(out_fake)
        loss_fake = ops.binary_cross_entropy_with_logits(out_fake, label_fake)
        return loss_real + loss_fake

    def forward_pass_gen(batch_size):
        # Generator wants the Discriminator to think fake images are REAL (1)
        fake = gen(batch_size)
        out_fake = ds(fake)
        label_real = ops.ones_like(out_fake)
        return ops.binary_cross_entropy_with_logits(out_fake, label_real)

    grad_fn_ds = ms.value_and_grad(forward_pass_ds, None, optimizer_ds.parameters)
    grad_fn_gen = ms.value_and_grad(forward_pass_gen, None, optimizer_gen.parameters)

    def train_step(real, train_discriminator):
        loss_ds = None
        if train_discriminator:
            loss_ds, grads_ds = grad_fn_ds(real)
            optimizer_ds(grads_ds)

        loss_gen, grads_gen = grad_fn_gen(real.shape[0])
        optimizer_gen(grads_gen)
        return loss_ds, loss_gen

    ratio = 10

    print("Starting Training...")
    for epoch in range(20):
        total_loss_ds = 0.0
        total_loss_gen = 0.0
        step_count = 0
        for idx, (data, label) in enumerate(fashion_ds.create_tuple_iterator()):
            train_discriminator = False
            if not idx % ratio:
                train_discriminator = True
            loss_ds, loss_gen = train_step(data, train_discriminator)
            if train_discriminator:
                total_loss_ds += loss_ds.asnumpy() * ratio
            total_loss_gen += loss_gen.asnumpy()
            step_count += 1

        avg_loss_ds = total_loss_ds / step_count
        avg_loss_gen = total_loss_gen / step_count
        print(
            f"Epoch {epoch + 1}: Discriminator Average Loss = {avg_loss_ds:.4f}: Generator Average Loss = {avg_loss_gen:.4f}"
        )

    fake_generated = gen(1)
    fake_label = Discriminator(fake_generated)

    print(f"Predicted label: {fake_label}")
    plt.imshow(fake_generated[0], cmap="grey")
    plt.show()
