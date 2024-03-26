from argparse import ArgumentParser
import os

import numpy as np

from common_utils import mkdir, reading, writing


def main():
    description = 'missing data generator'
    parser = ArgumentParser(description=description, add_help=False)
    parser.add_argument('--hap', type=str, required=True,
                        dest='hap_file', help='hap file')
    parser.add_argument('--legend', type=str, required=True,
                        dest='legend_file', help='legend file')
    parser.add_argument('--missing-rate', type=float, default=0.1,
                        dest='missing_rate', help='missing rate')
    parser.add_argument('--output-prefix', type=str, required=True,
                        dest='output_prefix', help='output prefix')
    parser.add_argument('--random-seed', type=int, default=3141692653,
                        dest='random_seed', help='random seed')
    args = parser.parse_args()

    num_variants = 0
    with reading(args.legend_file) as fin:
        fin.readline()
        for line in fin:
            num_variants += 1
    missing_size = int(args.missing_rate * num_variants)
    np.random.seed(args.random_seed)
    missing_indexes = np.random.choice(
        num_variants, size=missing_size, replace=False)
    missing_index_set = set(missing_indexes)
    mkdir(os.path.dirname(args.output_prefix))
    with reading(args.hap_file) as fin, \
         writing(args.output_prefix + '.hap.gz') as fout:
        for i, line in enumerate(fin):
            if i in missing_index_set:
                continue
            fout.write(line)
    with reading(args.legend_file) as fin, \
         writing(args.output_prefix + '.legend.gz') as fout:
        fout.write(fin.readline())
        for i, line in enumerate(fin):
            if i in missing_index_set:
                continue
            fout.write(line)


if __name__ == '__main__':
    main()
