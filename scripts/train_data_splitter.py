from argparse import ArgumentParser
import gc
import gzip
import math
import os
import sys

from common_utils import mkdir, reading, writing


class HapHandler():

    def __init__(self, hap_file):
        _, ext = os.path.splitext(hap_file)
        if ext == '.gz':
            self.fp = gzip.open(hap_file, 'rt')
        else:
            self.fp = open(hap_file, 'rt')
        self.max_index_in_line_buffer = -1
        self.min_index_in_line_buffer = -1
        self.line_buffer = []

    def get_line(self, index):
        assert index >= self.min_index_in_line_buffer, (
            'index {:s} must be >= min_index_in_line_buffer {:s}'.format(
                index, self.min_index_in_line_buffer))
        assert index <= self.max_index_in_line_buffer, (
            'index {:s} must be <= max_index_in_line_buffer {:s}'.format(
                index, self.max_index_in_line_buffer))
        return self.line_buffer[index - self.min_index_in_line_buffer]

    def load_to_buffer(self, min_index, max_index):
        assert min_index >= self.min_index_in_line_buffer, (
            'index {:d} must be >= min_index_in_line_buffer {:d}'.format(
                min_index, self.min_index_in_line_buffer))

        if max_index < self.max_index_in_line_buffer:
            max_index = self.max_index_in_line_buffer

        new_line_buffer = [None] * (max_index - min_index + 1)
        new_line_buffer_count = 0
        if min_index <= self.max_index_in_line_buffer:
            for index in range(min_index, self.max_index_in_line_buffer + 1):
                line = self.line_buffer[index - self.min_index_in_line_buffer]
                new_line_buffer[new_line_buffer_count] = line
                new_line_buffer_count += 1
            min_index = self.max_index_in_line_buffer + 1
        else:
            for _ in range(self.max_index_in_line_buffer + 1, min_index):
                self.fp.readline()

        self.line_buffer = new_line_buffer
        self.max_index_in_line_buffer = max_index
        self.min_index_in_line_buffer = (
            self.max_index_in_line_buffer - len(self.line_buffer) + 1)
        gc.collect()
        for _ in range(min_index, max_index + 1):
            line = self.fp.readline().rstrip()
            self.line_buffer[new_line_buffer_count] = line
            new_line_buffer_count += 1
        assert new_line_buffer_count == len(new_line_buffer)

    def close(self):
        self.fp.close()


def prepare_modified_hap(
        hap_file,
        legend_file,
        suppress_allele_flip,
        output_prefix,
        ):
    mkdir(os.path.dirname(output_prefix))
    a1_freq_list = []
    swap_flag_list = []
    skip_flag_list = []
    array_marker_flag_list = []
    with reading(legend_file) as fin:
        items = fin.readline().rstrip().split()
        try:
            array_marker_flag_col = items.index('array_marker_flag')
        except ValueError:
            print(
                'header \'array_marker_flag\' not found in ' + legend_file,
                file=sys.stderr)
            sys.exit(0)
        for line in fin:
            items = line.rstrip().split()
            array_marker_flag = items[array_marker_flag_col] == '1'
            array_marker_flag_list.append(array_marker_flag)
    with reading(hap_file) as fin, \
         writing(output_prefix + '.hap.gz') as fout:
        for i, line in enumerate(fin):
            if not array_marker_flag_list[i]:
                continue
            items = line.rstrip().split()
            a0_count = items.count('0')
            a1_count = items.count('1')
            swap_flag = a0_count < a1_count and array_marker_flag_list[i]
            if suppress_allele_flip:
                swap_flag = False
            swap_flag_list.append(swap_flag)
            if swap_flag:
                for j, item in enumerate(items):
                    if item == '0':
                        items[j] = '1'
                    elif item == '1':
                        items[j] = '0'
            a1_freq = 0
            if a0_count + a1_count > 0:
                if swap_flag:
                    a1_freq = a0_count / float(a0_count + a1_count)
                else:
                    a1_freq = a1_count / float(a0_count + a1_count)
            a1_freq_list.append(str(a1_freq))
            fout.write(' '.join(items))
            fout.write('\n')
    with reading(legend_file) as fin, \
         writing(output_prefix + '.legend.gz') as fout:
        header = fin.readline().rstrip()
        items = header.split()
        try:
            a0_col = items.index('a0')
            a1_col = items.index('a1')
        except ValueError:
            print(
                'some header item not found in ' + legend_file,
                file=sys.stderr)
            sys.exit(0)
        a1_freq_col = items.index('a1_freq') if 'a1_freq' in items else None
        swap_col = items.index('swap') if 'swap' in items else None
        fout.write(header)
        if a1_freq_col is None:
            fout.write(' a1_freq')
        if swap_col is None:
            fout.write(' swap')
        fout.write('\n')
        array_marker_count = 0
        for i, line in enumerate(fin):
            if not array_marker_flag_list[i]:
                continue
            items = line.rstrip().split()
            swap_flag = swap_flag_list[array_marker_count]
            if swap_flag:
                items[a0_col], items[a1_col] = items[a1_col], items[a0_col]
            a1_freq = a1_freq_list[array_marker_count]
            if a1_freq_col is None:
                items.append(a1_freq)
            else:
                items[a1_freq_col] = a1_freq
            if swap_col is None:
                items.append('1' if swap_flag else '0')
            else:
                items[swap_col] = '1' if swap_flag else '0'
            fout.write(' '.join(items))
            fout.write('\n')
            array_marker_count += 1


def load_partition_position_list(partition_file=None):
    partition_position_list = [1]
    if partition_file is None:
        return partition_position_list
    with open(partition_file, 'rt') as fin:
        fin.readline()
        for line in fin:
            _, _, stop = line.rstrip().split()
            partition_position_list.append(int(stop) + 1)
    return partition_position_list


def load_legend_info_list(legend_file):
    legend_info_list = []
    with reading(legend_file) as fin:
        items = fin.readline().rstrip().split(' ')
        try:
            position_col = items.index('position')
        except ValueError:
            print(
                'header \'position\' not found in ' + legend_file,
                file=sys.stderr)
            sys.exit(0)
        for line in fin:
            items = line.rstrip().split(' ')
            position = int(items[position_col])
            legend_info_list.append({
                'position': position,
            })
    return legend_info_list


def split_legend_info_list(legend_info_list, partition_position_list):
    if partition_position_list is None:
        return [0, len(legend_info_list) - 1]
    if len(partition_position_list) <= 1:
        return [0, len(legend_info_list) - 1]
    assert partition_position_list[-1] > legend_info_list[-1]['position']
    index_pair_list = []
    index = 0
    for partition_position in partition_position_list[1:]:
        start_index = index
        for legend_info in legend_info_list[start_index:]:
            if legend_info['position'] >= partition_position:
                break
            index += 1
        if index > start_index:
            index_pair_list.append([start_index, index - 1])
        if index >= len(legend_info_list):
            break
    assert index_pair_list[-1][1] == len(legend_info_list) - 1
    return index_pair_list


def split_local_legend_info_list(
        legend_info_list,
        marker_count_limit,
        index_origin):
    num_markers = len(legend_info_list)
    num_partitions = math.ceil(num_markers / marker_count_limit)
    num_markers_list = [int(num_markers / num_partitions)] * num_partitions
    remainder = num_markers - sum(num_markers_list)
    for i in range(remainder):
        num_markers_list[i] += 1
    start_index = 0
    region_info_list = []
    for num_markers in num_markers_list:
        end_index = start_index + num_markers - 1
        start_position = legend_info_list[start_index]['position']
        end_position = legend_info_list[end_index]['position']
        region_info_list.append({
            'start_index': start_index + index_origin,
            'end_index': end_index + index_origin,
            'start_position': start_position,
            'end_position': end_position,
        })
        start_index = end_index + 1
    return region_info_list


def get_region_info_list(
        legend_info_list,
        marker_count_limit,
        partition_position_list,
        ):
    partition_position_list[-1] = (
        legend_info_list[-1]['position'] + 1)
    index_pair_list = split_legend_info_list(
        legend_info_list, partition_position_list)
    region_info_list = []
    for start_index, end_index in index_pair_list:
        region_info_list.extend(
            split_local_legend_info_list(
                legend_info_list[start_index: end_index + 1],
                marker_count_limit, start_index))
    return region_info_list


def main():
    description = 'train data splitter'
    parser = ArgumentParser(description=description, add_help=False)
    parser.add_argument(
        '-h', '--hap', type=str, dest='hap_file', required=True,
        help='hap file')
    parser.add_argument(
        '-l', '--legend', type=str, dest='legend_file', required=True,
        help='legend file')
    parser.add_argument(
        '-p', '--partition', type=str, dest='partition_file', default=None,
        help='partition file')
    parser.add_argument(
        '-o', '--output-prefix', type=str, dest='output_prefix',
        required=True, help='output prefix')
    parser.add_argument(
        '-r', '--marker-count-limit', type=int,
        dest='marker_count_limit', default=10000,
        help='marker count limit')
    parser.add_argument(
        '--suppress-allele-flip', action='store_true', default=False,
        dest='suppress_allele_flip', help='suppress allele flip')
    args = parser.parse_args()

    prepare_modified_hap(
        args.hap_file, args.legend_file, args.suppress_allele_flip,
        args.output_prefix + '_mod')
    partition_position_list = load_partition_position_list(args.partition_file)
    legend_info_list = load_legend_info_list(
        args.output_prefix + '_mod.legend.gz')
    region_info_list = get_region_info_list(
        legend_info_list, args.marker_count_limit, partition_position_list)

    print('No. of splitted regions: {:d}'.format(len(region_info_list)))

    hap_handler = HapHandler(args.output_prefix + '_mod.hap.gz')
    legend_lines = []
    with reading(args.output_prefix + '_mod.legend.gz') as fin:
        legend_header = fin.readline()
        for line in fin:
            legend_lines.append(line)
    for i, region_info in enumerate(region_info_list, start=1):
        start_index = region_info['start_index']
        end_index = region_info['end_index']
        hap_handler.load_to_buffer(start_index, end_index)
        output_file = '{:s}_{:d}.hap.gz'.format(args.output_prefix, i)
        print(output_file, flush=True)
        with writing(output_file) as fout:
            for index in range(start_index, end_index + 1):
                fout.write(hap_handler.get_line(index))
                fout.write('\n')
        output_file = '{:s}_{:d}.legend.gz'.format(args.output_prefix, i)
        with writing(output_file) as fout:
            fout.write(legend_header)
            for index in range(start_index, end_index + 1):
                fout.write(legend_lines[index])
    hap_handler.close()

    mkdir(os.path.dirname(args.output_prefix))
    with open(args.output_prefix + '_region_info.txt', 'wt') as fout:
        line = '\t'.join([
            'region_id',
            'start_pos',
            'end_pos',
            'marker_count',
        ])
        fout.write(line)
        fout.write('\n')
        for i, region_info in enumerate(region_info_list, start=1):
            start_index = region_info['start_index']
            end_index = region_info['end_index']
            marker_count = end_index - start_index + 1
            line = '\t'.join(map(str, [
                i,
                region_info['start_position'],
                region_info['end_position'],
                marker_count,
            ]))
            fout.write(line)
            fout.write('\n')
    with open(args.output_prefix + '.list', 'wt') as fout:
        for i in range(1, len(region_info_list) + 1):
            fout.write('{:s}_{:d}\n'.format(args.output_prefix, i))


if __name__ == '__main__':
    main()
