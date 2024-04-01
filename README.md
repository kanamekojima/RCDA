# RCDA

SNP array genotype data often contains missing values due to experimental errors.
Additionally, genotypes for some markers are removed during the quality check process if they deviate from Hardy-Weinberg equilibrium or exhibit a high missing rate.
Since some reference-free genotype imputation methods such as [RNN-IMP](https://doi.org/10.1371/journal.pcbi.1008207) and [HIBAG](https://doi.org/10.1038/tpj.2013.18) are not robust against missing values in input SNP array genotype data, there is a significant demand for reference-free missing imputation methods that can overcome these drawbacks while retaining the benefits of being reference-free.

RCDA is a Python program designed to address the limitations of these methods by imputing missing genotypes in SNP array genotype data in a reference-free manner, using Residual Convolutional Denoising Autoencoders.
It accepts phased genotypes in HAPSLEGEND format as input and outputs the imputed genotypes in either VCF or HAPSLEGEND format.
The drawbacks of RNN-IMP and HIBAG can be alleviated by using the imputed genotypes generated by RCDA as their input data.

## Installation

Requirements: Python versions 3.5 to 3.10 (ensure python3 is in your path)

```sh
git clone https://github.com/kanamekojima/RCDA.git
cd RCDA
python3 -m pip install -r requirements.txt
```

## Example Usage

### Preparation of Example Dataset

To prepare the example dataset, the following files are required:

- A VCF file from the 1000 Genomes Project (1KGP) phase 3 dataset for chromosome 22 (`ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz`):
  - Download the VCF file from the following website and place it in the `org_data` directory:
    - [https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502](https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502)
- A manifest file for the Infinium Omni2.5-8 BeadChip (`InfiniumOmni2-5-8v1-4_A1.csv`):
  - Download `infinium-omni-2-5-8v1-4-a1-manifest-file-csv.zip` from the following website:
    - [https://support.illumina.com/array/array_kits/humanomni2_5-8_beadchip_kit/downloads.html](https://support.illumina.com/array/array_kits/humanomni2_5-8_beadchip_kit/downloads.html)
    - Unzip the file and place `InfiniumOmni2-5-8v1-4_A1.csv` in the `org_data` directory.
- An Hg19 fasta file (hg19.fa):
  - Download `hg19.fa.gz` from the following website:
    - [https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/](https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/)
    - Unzip the file and place `hg19.fa` in the `org_data` directory.

After preparing the above files, execute the following command in the `RCDA` directory:

```sh
python3 scripts/test_data_preparation.py
```

This process generates the example dataset, including:

- `example_data/test/chr22_true.[hap.gz/legend.gz]`: Phased genotype data of 100 individuals from `org_data/test_samples.txt` for marker sites designed for the Infinium Omni2.5-8 BeadChip on chromosome 22, in HAPSLEGEND format. This data is derived from the 1KGP phase 3 dataset. These 100 individuals are randomly selected from the 2,504 individuals in the 1KGP phase 3 dataset.
- `example_data/train/chr22.[hap.gz/legend.gz]`: Phased genotype data of the remaining 2,404 individuals not included in `org_data/test_samples.txt` for marker sites designed for the Infinium Omni2.5-8 BeadChip on chromosome 22, in HAPSLEGEND format. This dataset is also derived from the 1KGP phase 3 dataset and used for training RCDA.

Missing genotype data for testing RCDA can be generated from `example_data/test/chr22_true.[hap.gz/legend.gz]` with the following command in `RCDA` direcotry:

```sh
for missing_rate in 0.01 0.05 0.1 0.2
do
  python3 scripts/missing_data_generator.py \
    --hap example_data/test/chr22_true.hap.gz \
    --legend example_data/test/chr22_true.legend.gz \
    --missing-rate $missing_rate \
    --output-prefix example_data/test/chr22_${missing_rate}
done
```

### Imputation for the Example Dataset

RCDA performs imputation on small regions separately and combines these results to produce missing imputed genotypes for an entire chromosome in VCF format or HAPSLEGND format.
For each small region, a specific model structure and its parameters, stored in the ONNX Runtime (ORT) format, along with target information in the legend format, are required.
ORT files and legend files from the example training data are located in the `results/train` directory.
These files can also be generated through the training process, as described in the subsequent section.
To perform imputation on the missing genotype data using the aforementioned model information files, execute the following command in the `RCDA` directory:

```sh
python3 scripts/imputation.py \
    --hap example_data/test/chr22_0.2.hap.gz \
    --legend example_data/test/chr22_0.2.legend.gz \
    --model-prefix results/train/chr22 \
    --output-prefix results/imputation/chr22_0.2
```

This command generates the missing imputed results for `example_data/test/chr22_0.2.[hap.gz/legend.gz]` as `results/imputation/chr22_0.2.[hap.gz/legend.gz]`.
To produce the results in VCF format, use the `--output-format vcf` option.

### Training Models for the Example Dataset

In RCDA, the whole chromosome is divided into small regions, and deep learning (DL) models are trained separately for each region.
To begin, divide the training data, `example_data/train/chr22.[hap.gz/legend.gz]`, into smaller segments using the following commands in the `RCDA` directory:

```sh
wget https://raw.githubusercontent.com/stephenslab/ldshrink/main/inst/test_gdsf/fourier_ls-all.bed -P org_data
head -n 1 org_data/fourier_ls-all.bed > org_data/fourier_ls-chr22.bed
grep "^chr22 " org_data/fourier_ls-all.bed >> org_data/fourier_ls-chr22.bed
python3 scripts/train_data_splitter.py \
    --hap example_data/train/chr22.hap.gz \
    --legend example_data/train/chr22.legend.gz \
    --output-prefix example_data/train/split/chr22 \
    --marker-count-limit 500 \
    --suppress-allele-flip \
    --partition org_data/fourier_ls-chr22.bed
```

The `--partition` option specifies a file containing a list of regions into which the chromosome is divided, facilitating segmentation based on these predefined regions.
For this example, a list of regions segmented at high recombination rate points is used.
This list is available on the Stephens lab GitHub page as a file named `fourier_ls-all.bed`:
[https://github.com/stephenslab/ldshrink](https://github.com/stephenslab/ldshrink)

Based on the specified criteria, these segmented regions are further divided in the above commands, resulting in the generation of training data files for 75 divided regions in this example.
The prefixes for these files are listed in `example_data/train/split/chr22.list`.

To train DL models for these regions using the segmented training data, execute the following command in the `RCDA` directory:

```sh
python3 scripts/train.py \
    --data-list example_data/train/split/chr22.list \
    --model-type ResNet \
    --epochs 1000 \
    --dropout-rate 0 \
    --learning-rate 0.001 \
    --output-prefix results/train/chr22
```

## Options

### Options for `scripts/imputation.py`

| Option | Default Value | Summary |
|:-------|:-------------:|:-------|
| --hap STRING_VALUE | - | Input hap file |
| --legend STRING_VALUE | - | Input legend file |
| --sample STRING_VALUE | None | Input sample file (optional) |
| --chromosome | None | Chromosome name (required for VCF output format) |
| --model-prefix STRING_VALUE | - | Model name prefix |
| --output-prefix STRING_VALUE | - | Output file name prefix |
| --output-format STRING_VALUE | gen | Output format [gen / vcf] |
| --python3-bin STRING_VALUE | python3 | Path to the Python3 binary |

Options for `scripts/train.py`

| Option | Default Value | Summary |
|:-------|:-------------:|:-------|
| --data-list STRING_VALUE | - | Input list file |
| --output-prefix STRING_VALUE | - | Output file name prefix |
| --model-type STRING_VALUE | ResNet | DL model type (available options: ResNet / ResNetV2 / SCDA) |
| --epochs INT_VALUE | 1000 | Number of epochs |
| --batch-size INT_VALUE | 32 | Training batch size |
| --dropout-rate FLOAT_VALUE | 0 | Dropout rate |
| --learning-rate FLOAT_VALUE | 0.001 | Learning rate |
| --monitor STRING_VALUE | val_loss | Early stopping criterion |
| --missing-rate-interval STRING_VALUE | 0.2:0.5 | Missing rate interval applied to training data |
| --validation-missing-rate FLOAT_VALUE | 0.25 | Missing rate in validation data |
| --validation-sample-size INT_VALUE | 100 | Validation sample size |
| --data-augmentation | False | Data augmentation is applied |
| --python3-bin STRING_VALUE | python3 | Path to the Python3 binary |

## License

The scripts in this repository are available under the MIT License.
For more details, see the [LICENSE](LICENSE) file.

## Contact

Developer: Kaname Kojima, Ph.D.

E-mail: kojima [AT] megabank [DOT] tohoku [DOT] ac [DOT] jp or kengo [AT] ecei [DOT] tohoku [DOT] ac [DOT] jp
