from argparse import ArgumentParser
import os
import sys
import time

from common_utils import system


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    description = 'train'
    parser = ArgumentParser(description=description, add_help=False)
    parser.add_argument('--data-list', type=str, required=True,
                        dest='data_prefix_list_file',
                        help='file for data file prefix list')
    parser.add_argument('--output-prefix', type=str, required=True,
                        dest='output_prefix', help='output prefix')
    parser.add_argument('--model-type', type=str, default='SCDA',
                        dest='model_type', help='model type')
    parser.add_argument('--epochs', type=int, default=10,
                        dest='epochs', help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        dest='batch_size', help='batch size')
    parser.add_argument('--dropout-rate', type=float, default=0.25,
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
    parser.add_argument('--python3-bin', type=str, default='python3',
                        dest='python3_bin', help='path to Python3 binary')
    args = parser.parse_args()

    start_time = time.time()
    data_prefix_list = []
    with open(args.data_prefix_list_file) as fin:
        for line in fin:
            data_prefix = os.path.join(
                os.path.dirname(args.data_prefix_list_file),
                os.path.basename(line.strip()))
            data_prefix_list.append(data_prefix)
    common_options = [
        '--model-type ' + args.model_type,
        '--epochs {:d}'.format(args.epochs),
        '--batch-size {:d}'.format(args.batch_size),
        '--dropout-rate {:f}'.format(args.dropout_rate),
        '--learning-rate {:f}'.format(args.learning_rate),
        '--monitor ' + args.monitor,
        '--missing-rate-interval ' + args.missing_rate_interval,
        '--validation-missing-rate {:f}'.format(args.validation_missing_rate),
        '--validation-sample-size {:d}'.format(args.validation_sample_size),
    ]
    if args.data_augmentation:
        common_options.append('--data-augmentation')
    for i, data_prefix in enumerate(data_prefix_list, start=1):
        previous_job_id_list = []
        output_prefix = '{:s}_{:d}'.format(args.output_prefix, i)
        options = [
            '--hap {:s}.hap.gz'.format(data_prefix),
            '--legend {:s}.legend.gz'.format(data_prefix),
            '--output-prefix ' + output_prefix,
        ]
        command_list = []
        command = args.python3_bin
        command += ' ' + os.path.join(SCRIPT_DIR, 'train_model.py')
        command += ' ' + ' '.join(options + common_options)
        command_list.append(command)

        command = args.python3_bin
        command += ' -m tf2onnx.convert'
        command += ' --saved-model ' + output_prefix
        command += ' --rename-inputs inputs'
        command += ' --rename-outputs outputs'
        command += ' --output {:s}.onnx'.format(output_prefix)
        command_list.append(command)

        command = args.python3_bin
        command += ' -m onnxruntime.tools.convert_onnx_models_to_ort'
        command += ' {:s}.onnx'.format(output_prefix)
        command_list.append(command)

        command = '; '.join(command_list) + ';'
        system(command)

    elapsed_time = time.time() - start_time
    with open(args.output_prefix + '.time', 'wt') as fout:
        fout.write('Elapsed time: {:f} [s]\n'.format(elapsed_time))


if __name__ == '__main__':
    main()
