from argparse import ArgumentParser
import os
import sys

from common_utils import mkdir, reading, writing
from hapslegend_utils import hapslegend2haps, haps2hapslegend


def merge(haps_file1, haps_file2, output_file):

    def get_position(line):
        space_count = 0
        previous_space_position = -1
        for i, c in enumerate(line):
            if c == ' ':
                space_count += 1
                if space_count == 2:
                    return int(line[previous_space_position + 1 : i])
                previous_space_position = i
        return int(line[previous_space_position + 1 :])

    with reading(haps_file1) as fin1, \
         reading(haps_file2) as fin2, \
         writing(output_file) as fout:
        position2 = -1
        line2 = None
        for line1 in fin1:
            position1 = get_position(line1)
            if position1 > position2:
                if line2 is not None:
                    fout.write(line2)
                    line2 = None
                for line2 in fin2:
                    position2 = get_position(line2)
                    if position1 <= position2:
                        break
                    fout.write(line2)
                    line2 = None
            fout.write(line1)
        if line2 is not None:
            fout.write(line2)
        for line2 in fin2:
            fout.write(line2)


def main():
    description = 'merge hapslegend and haps'
    parser = ArgumentParser(description=description, add_help=False)
    parser.add_argument('--hap', type=str,
                        dest='hap_file', required=True,
                        help='hap file')
    parser.add_argument('--legend', type=str,
                        dest='legend_file', required=True,
                        help='legend file')
    parser.add_argument('--haps-prefix', type=str,
                        dest='haps_prefix', required=True,
                        help='haps prefix')
    parser.add_argument('--start-index', type=int,
                        dest='start_index', required=True, help='start index')
    parser.add_argument('--end-index', type=int,
                        dest='end_index', required=True, help='start index')
    parser.add_argument('--output-prefix', type=str,
                        dest='output_prefix', required=True,
                        help='output prefix')
    args = parser.parse_args()

    mkdir(os.path.dirname(args.output_prefix))
    hapslegend2haps(
        args.hap_file, args.legend_file, args.output_prefix + '.tmp1.gz')
    with writing(args.output_prefix + '.tmp2.gz') as fout:
        for index in range(args.start_index, args.end_index + 1):
            haps_file = '{:s}_{:d}.haps.gz'.format(args.haps_prefix, index)
            if not os.path.exists(haps_file):
                print('Warning: {:s} does not exist'.format(haps_file))
                continue
            with reading(haps_file) as fin:
                for line in fin:
                    fout.write(line)
    merge(
        args.output_prefix + '.tmp1.gz',
        args.output_prefix + '.tmp2.gz',
        args.output_prefix + '.tmp3.gz',
        )
    os.remove(args.output_prefix + '.tmp1.gz')
    os.remove(args.output_prefix + '.tmp2.gz')
    haps2hapslegend(args.output_prefix + '.tmp3.gz', args.output_prefix)
    os.remove(args.output_prefix + '.tmp3.gz')


if __name__ == '__main__':
    main()
