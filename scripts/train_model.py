from argparse import ArgumentParser
import os
import shutil
import sys
import time

import numpy as np
from tensorflow import keras

from common_utils import mkdir, reading
import DL_models


class TrainDataGenerator(keras.utils.Sequence):

    def __init__(
            self,
            x_dataset,
            batch_size,
            left_margin,
            right_margin,
            missing_rate_interval=[0.05, 0.25],
            data_augmentation=None,
            shuffle=True,
            ):
        self.x = x_dataset
        self.batch_size = batch_size
        self.missing_rate_interval = missing_rate_interval
        self.shuffle = shuffle
        self.left_index = left_margin
        self.right_index = self.x.shape[1] - right_margin
        self.data_augmentation = data_augmentation
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.x.shape[0] / self.batch_size))

    def __getitem__(self, index):
        batch_start_index = index * self.batch_size
        batch_end_index = batch_start_index + self.batch_size
        indexes = self.indexes[batch_start_index: batch_end_index]
        inputs = self.inputs[indexes]
        outputs = self.outputs[indexes]

        for i in range(inputs.shape[0]):
            missing_rate = np.random.uniform(*self.missing_rate_interval)
            missing_size = int(missing_rate * inputs.shape[1])
            missing_index = np.random.choice(
                inputs.shape[1], size=missing_size, replace=False)
            missing_index.sort()
            inputs[i, missing_index, :] = [1, 0, 0]
        return inputs, outputs

    def on_epoch_end(self):
        if self.data_augmentation is None:
            self.outputs = self.x
        else:
            self.outputs = self.data_augmentation.get_augmented_data()
        self.indexes = np.arange(self.outputs.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        self.inputs = self.outputs.copy()
        self.outputs = self.outputs[:, self.left_index:self.right_index, 1:]


class ValidationDataGenerator(keras.utils.Sequence):

    def __init__(
            self,
            x_dataset,
            batch_size,
            left_margin,
            right_margin,
            missing_rate,
            ):
        self.x = x_dataset
        self.batch_size = batch_size
        self.left_index = left_margin
        self.right_index = self.x.shape[1] - right_margin
        self.missing_rate = missing_rate
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.x.shape[0] / self.batch_size))

    def __getitem__(self, index):
        batch_start_index = index * self.batch_size
        batch_end_index = batch_start_index + self.batch_size
        indexes = self.indexes[batch_start_index: batch_end_index]
        inputs = self.inputs[indexes]
        outputs = self.outputs[indexes]

        missing_size = int(self.missing_rate * inputs.shape[1])
        for i in range(inputs.shape[0]):
            missing_index = np.random.choice(
                inputs.shape[1], size=missing_size, replace=False)
            missing_index.sort()
            inputs[i, missing_index, :] = [1, 0, 0]
        return inputs, outputs

    def on_epoch_end(self):
        self.indexes = np.arange(self.x.shape[0])
        self.outputs = self.x
        self.inputs = self.outputs.copy()
        self.outputs = self.outputs[:, self.left_index:self.right_index, 1:]


class DataAugmentation:

    def __init__(
            self,
            haplotypes,
            positions,
            mutation_rate=1.1e-8,
            recombination_rate=1.0e-8,
            Ne=1e4,
            block_size=2,
            num_recombination_points=1,
            ):
        self.haplotypes = haplotypes
        self.block_size = block_size
        self.num_recombination_points = num_recombination_points
        self.flip_prob = self.get_flip_prob(len(haplotypes), mutation_rate, Ne)
        r = len(haplotypes) % self.block_size
        if r == 0:
            self.augmented_haplotypes = np.zeros_like(haplotypes)
        else:
            self.augmented_haplotypes = np.zeros_like(haplotypes[:-r])
        self.recombination_prob_dist = self.get_recombination_probs(
            positions, self.block_size, recombination_rate, Ne)
        self.recombination_prob_dist /= self.recombination_prob_dist.sum()

    def get_flip_prob(self, num_haplotypes, mutation_rate, Ne):
        theta = 4.0 * Ne * mutation_rate
        mutation_prob = 0.5 * theta / (theta + num_haplotypes)
        return mutation_prob

    def get_recombination_probs(self, positions, k, recombination_rate, Ne):
        distances = positions[1:] - positions[:-1]
        rho = 4 * Ne * recombination_rate
        recombination_probs = 1.0 - np.exp(- distances * rho / k)
        return recombination_probs

    def get_augmented_data(self):

        def exclusive_or(x, y):
            return x + y - 2 * x * y

        recombination_points = np.zeros(
            self.num_recombination_points + 1, np.int64)
        recombination_points[-1] = self.haplotypes.shape[1]
        r_indexes = np.arange(self.haplotypes.shape[0])
        s_indexes = np.arange(self.block_size)
        np.random.shuffle(r_indexes)
        for i in range(len(self.augmented_haplotypes) // self.block_size):
            if self.num_recombination_points >= 1:
                recombination_points[:-1] = np.random.choice(
                    len(self.recombination_prob_dist),
                    self.num_recombination_points, replace=False,
                    p=self.recombination_prob_dist) + 1
                recombination_points.sort()
                assert recombination_points[-1] == self.haplotypes.shape[1]
            p = 0
            for p_next in recombination_points:
                np.random.shuffle(s_indexes)
                for j, s_index in enumerate(s_indexes):
                    index = self.block_size * i + j
                    augmented_haplotype = self.augmented_haplotypes[index]
                    index = r_indexes[self.block_size * i + s_index]
                    haplotype = self.haplotypes[index]
                    augmented_haplotype[p:p_next] = haplotype[p:p_next]
                p = p_next
        flips = np.random.rand(
            *self.augmented_haplotypes.shape[:2]) < self.flip_prob
        flips = flips.astype(self.augmented_haplotypes.dtype)
        flips = np.expand_dims(flips, -1)
        self.augmented_haplotypes[:, :, 1:] = exclusive_or(
            flips, self.augmented_haplotypes[:, :, 1:])
        return self.augmented_haplotypes


def load_data(hap_file, legend_file):
    positions = []
    a1_freqs = []
    with reading(legend_file) as fin:
        header_items = fin.readline().rstrip().split(' ')
        try:
            a1_freq_col = header_items.index('a1_freq')
            position_col = header_items.index('position')
        except ValueError:
            print(
                'Some header items not found in ' + legend_file,
                file=sys.stderr)
            sys.exit(0)
        for line in fin:
            items = line.rstrip().split(' ')
            positions.append(int(items[position_col]))
            a1_freqs.append(float(items[a1_freq_col]))
    with reading(hap_file) as fin:
        for i, line in enumerate(fin):
            items = line.rstrip().split(' ')
            if i == 0:
                inputs = np.zeros([len(items), len(positions), 3])
                inputs[:, :, 0] = 1
            for j, item in enumerate(items):
                if item == '0':
                    inputs[j, i] = [0, 1, 0]
                elif item == '1':
                    inputs[j, i] = [0, 0, 1]
        return inputs, positions, a1_freqs


def get_random_indexes(num_samples, seed):
    ordering = np.arange(num_samples)
    np.random.seed(seed)
    np.random.shuffle(ordering)
    return np.stack(
        [2 * ordering, 2 * ordering + 1], axis=1).reshape(-1)


def main():
    description = 'train'
    parser = ArgumentParser(description=description, add_help=False)
    parser.add_argument('--hap', type=str, required=True, dest='hap_file',
                        help='hap file')
    parser.add_argument('--legend', type=str, required=True,
                        dest='legend_file', help='legend file')
    parser.add_argument('--output-prefix', type=str, required=True,
                        dest='output_prefix', help='output prefix')
    parser.add_argument('--model-type', type=str, default='ResNet',
                        dest='model_type', help='model type')
    parser.add_argument('--epochs', type=int, default=1000,
                        dest='epochs', help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        dest='batch_size', help='batch size')
    parser.add_argument('--dropout-rate', type=float, default=0,
                        dest='dropout_rate', help='dropout rate')
    parser.add_argument('--learning-rate', type=float, default=1.e-3,
                        dest='learning_rate', help='learning rate')
    parser.add_argument('--monitor', type=str, default='val_loss',
                        dest='monitor', help='monitor')
    parser.add_argument('--missing-rate-interval', type=str, default='0.2:0.5',
                        dest='missing_rate_interval',
                        help='missing rate interval')
    parser.add_argument('--validation-missing-rate', type=float, default=0.25,
                        dest='validation_missing_rate',
                        help='validation missing rate')
    parser.add_argument('--validation-sample-size', type=int, default=100,
                        dest='validation_sample_size',
                        help='validation sample size')
    parser.add_argument('--data-augmentation', action='store_true',
                        default=False,
                        dest='data_augmentation', help='data augmentation')
    parser.add_argument('--random-seed', type=int, default=3141592653,
                        dest='random_seed', help='random seed')
    args = parser.parse_args()

    assert args.model_type in {'ResNet', 'ResNetV2', 'SCDA'}
    start_time = time.time()
    mkdir(os.path.dirname(args.output_prefix))
    shutil.copy(args.legend_file, args.output_prefix + '.legend.gz')
    inputs, positions, a1_freqs = load_data(args.hap_file, args.legend_file)
    left_margin = 0
    right_margin = (4 - (inputs.shape[1] % 4)) % 4
    if right_margin != 0:
        tail = np.tile(
            np.array([1, 0, 0], inputs.dtype), [len(inputs), right_margin, 1])
        inputs = np.concatenate([inputs, tail], 1)
        positions += [positions[-1]] * right_margin
    indexes = get_random_indexes(len(inputs) // 2, args.random_seed)
    inputs_train = inputs[indexes[:-2 * args.validation_sample_size]]
    inputs_val = inputs[indexes[-2 * args.validation_sample_size:]]

    missing_rate_interval = list(
        map(float, args.missing_rate_interval.split(':')))
    print(
        'Missing rate interval: {:.4f}-{:.4f}'.format(*missing_rate_interval))

    if args.model_type == 'ResNet':
        model = DL_models.get_ResNet_model(
            inputs.shape[-1], right_margin=right_margin,
            dropout_rate=args.dropout_rate)
    elif args.model_type == 'ResNetV2':
        model = DL_models.get_ResNetV2_model(
            inputs.shape[-1], right_margin=right_margin,
            dropout_rate=args.dropout_rate)
    elif args.model_type == 'SCDA':
        model = DL_models.get_SCDA_model(
            inputs.shape[-1], right_margin=right_margin,
            dropout_rate=args.dropout_rate)
    else:
        print('Unsupported model type: ' + args.model_type, file=sys.stderr)

    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(
        loss='categorical_crossentropy', optimizer=optimizer,
        metrics=['accuracy'])
    model.build(input_shape=(None, *inputs.shape[1:]))
    model.summary()

    if args.data_augmentation:
        data_augmentation = DataAugmentation(inputs_train, np.array(positions))
    else:
        data_augmentation = None
    train_data_generator = TrainDataGenerator(
        inputs_train, args.batch_size, left_margin, right_margin,
        missing_rate_interval, data_augmentation)
    validation_data_generator = ValidationDataGenerator(
        inputs_val, args.batch_size, left_margin, right_margin,
        missing_rate=0.25)
    ModelCheckpoint = keras.callbacks.ModelCheckpoint(
        args.output_prefix + '_best.h5',
        monitor=args.monitor,
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        period=1)
    model.fit_generator(
        generator=train_data_generator,
        validation_data=validation_data_generator,
        epochs=args.epochs,
        verbose=2,
        callbacks=[ModelCheckpoint])
    model.save_weights(args.output_prefix + '_last.h5')
    model.load_weights(args.output_prefix + '_best.h5')
    model.save(args.output_prefix)
    elapsed_time = time.time() - start_time
    with open(args.output_prefix + '.time', 'wt') as fout:
        fout.write('Elapsed time: {:f} [s]\n'.format(elapsed_time))


if __name__ == '__main__':
    main()
