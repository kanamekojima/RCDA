import os
import sys

from common_utils import mkdir, reading, writing


def hapslegend2haps(hap_file, legend_file, output_file):
    with reading(hap_file) as fin_hap, \
         reading(legend_file) as fin_legend, \
         writing(output_file) as fout:
        header_items = fin_legend.readline().rstrip().split(' ')
        try:
            id_col = header_items.index('id')
            a0_col = header_items.index('a0')
            a1_col = header_items.index('a1')
            position_col = header_items.index('position')
        except ValueError:
            print(
                'Some of header items not found in ' + legend_file,
                file=sys.stderr)
            sys.exit(0)
        for line in fin_hap:
            items = fin_legend.readline().rstrip().split(' ')
            a0 = items[a0_col]
            a1 = items[a1_col]
            snp_name = items[id_col]
            position = items[position_col]
            fout.write('{:s} {:s} {:s} {:s} {:s}'.format(
                snp_name, position, a0, a1, line))


def haps2hapslegend(haps_file, output_prefix):
    with reading(haps_file) as fin, \
         writing(output_prefix + '.hap.gz') as fout_hap, \
         writing(output_prefix + '.legend.gz') as fout_legend:
        fout_legend.write('id position a0 a1\n')
        for line in fin:
            items = line.rstrip().split(' ')
            space_count = 0
            for i, c in enumerate(line):
                if c == ' ':
                    space_count += 1
                if space_count == 4:
                    break
            fout_hap.write(line[i + 1:])
            fout_legend.write(line[:i])
            fout_legend.write('\n')


def write_haps(predictions, legend_record_list, output_file):
    predictions = predictions.argmax(-1)
    mkdir(os.path.dirname(output_file))
    with writing(output_file) as fout:
        for i, legend_record in enumerate(legend_record_list):
            fout.write('{:s} {:s} {:s} {:s} '.format(
                legend_record.id, legend_record.position,
                legend_record.a0, legend_record.a1))
            fout.write(' '.join(map(str, predictions[:, i])))
            fout.write('\n')
