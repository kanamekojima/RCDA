from argparse import ArgumentParser
from collections import namedtuple
import sys

import numpy as np
import onnxruntime

from common_utils import mkdir, reading
from vcf_utils import write_vcf
from hapslegend_utils import write_haps


LegendRecord = namedtuple(
    'LegendRecord', (
        'id',
        'position',
        'a0',
        'a1',
    )
)


def load_legend(legend_file):
    legend_record_list = []
    with reading(legend_file) as fin:
        items = fin.readline().strip().split()
        try:
            id_col = items.index('id')
            a0_col = items.index('a0')
            a1_col = items.index('a1')
            position_col = items.index('position')
        except ValueError:
            print(
                'Some header items not found in ' + legend_file,
                file=sys.stderr)
            sys.exit(0)
        for line in fin:
            items = line.strip().split()
            legend_record = LegendRecord(
                items[id_col],
                items[position_col],
                items[a0_col],
                items[a1_col],
            )
            legend_record_list.append(legend_record)
    return legend_record_list


def load_data(hap_file, legend_file, legend_record_list):
    num_variants = len(legend_record_list)
    num_samples = None
    with reading(hap_file) as fin:
        items = fin.readline().rstrip().split(' ')
        num_samples = len(items)
    assert num_samples is not None
    inputs = np.zeros([num_samples, num_variants, 3])
    inputs[:, :, 0] = 1

    observation_flag_list = [False] * num_variants
    variant_index_dict = {}
    for i, legend_record in enumerate(legend_record_list):
        position = legend_record.position
        a0, a1 = legend_record.a0, legend_record.a1
        key = ':'.join([position, a0, a1])
        variant_index_dict[key] = i

    with reading(legend_file) as legend_fin, \
         reading(hap_file) as hap_fin:
        header_items = legend_fin.readline().rstrip().split(' ')
        try:
            a0_col = header_items.index('a0')
            a1_col = header_items.index('a1')
            position_col = header_items.index('position')
        except ValueError:
            print(
                'Some header items not found in '+ legend_file,
                file=sys.stderr)
            sys.exit(0)
        for line in legend_fin:
            items = line.rstrip().split(' ')
            position = items[position_col]
            a0, a1 = items[a0_col], items[a1_col]
            key1 = ':'.join([position, a0, a1])
            key2 = ':'.join([position, a1, a0])
            line = hap_fin.readline()
            if key1 in variant_index_dict:
                variant_index = variant_index_dict[key1]
                flip_flag = False
            elif key2 in variant_index_dict:
                variant_index = variant_index_dict[key2]
                flip_flag = True
            else:
                continue
            assert flip_flag is False, items[0]
            observation_flag_list[variant_index] = True
            vector0, vector1 = [0, 1, 0], [0, 0, 1]
            if flip_flag:
                vector0, vector1 = vector1, vector0
            items = line.rstrip().split(' ')
            for i, item in enumerate(items):
                if item == '0':
                    inputs[i, variant_index] = vector0
                elif item == '1':
                    inputs[i, variant_index] = vector1
    missing_indexes = np.array([
        i for i, observation_flag in enumerate(observation_flag_list)
        if not observation_flag
    ])
    return inputs, missing_indexes


def predict(inputs, model_file):
    sess_options = onnxruntime.SessionOptions()
    sess_options.enable_cpu_mem_arena = False
    sess_options.enable_mem_pattern = False
    sess_options.enable_mem_reuse = False
    sess = onnxruntime.InferenceSession(
        model_file, sess_options, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    predictions = sess.run(None, {input_name: inputs})[0]
    return predictions


def main():
    description = 'inference'
    parser = ArgumentParser(description=description, add_help=False)
    parser.add_argument('--hap', type=str, required=True,
                        dest='hap_file', help='hap file')
    parser.add_argument('--legend', type=str, required=True,
                        dest='legend_file', help='legend file')
    parser.add_argument('--model-prefix', type=str, required=True,
                        dest='model_prefix', help='model file prefix')
    parser.add_argument('--output-file', type=str, required=True,
                        dest='output_file', help='output file')
    parser.add_argument('--output-format', type=str, default='haps',
                        dest='output_format', help='output format')
    args = parser.parse_args()

    legend_record_list = load_legend(args.model_prefix + '.legend.gz')
    inputs, missing_indexes = load_data(
        args.hap_file, args.legend_file, legend_record_list)
    if len(missing_indexes) == 0:
        predictions = np.zeros([len(inputs), 0, 2], dtype=np.float32)
    else:
        right_margin = (4 - (inputs.shape[1] % 4)) % 4
        if right_margin != 0:
            tail = np.tile(
                np.array([1, 0, 0], inputs.dtype),
                [len(inputs), right_margin, 1])
            inputs = np.concatenate([inputs, tail], 1)
        inputs = inputs.astype(np.float32)
        predictions = predict(inputs, args.model_prefix + '.ort')
        predictions = predictions[:, missing_indexes]
    missing_legend_record_list = [
        legend_record_list[i] for i in missing_indexes]
    if args.output_format == 'vcf':
        predictions = predictions.transpose(1, 0, 2)
        write_vcf(predictions, missing_legend_record_list, args.output_file)
    elif args.output_format == 'haps':
        write_haps(predictions, missing_legend_record_list, args.output_file)
    else:
        print('Unsupported format: ' + args.output_format, file=sys.stderr)


if __name__ == '__main__':
    main()
