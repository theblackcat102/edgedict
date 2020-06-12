import argparse
import os
import subprocess
import hashlib
import glob
import tarfile
import re

from tqdm import tqdm
from sphfile import SPHFile

# ========== ===========
# Parse input arguments
# ========== ===========
parser = argparse.ArgumentParser(description="LibriSpeech downloader")
parser.add_argument(
    '--save_path', type=str, default="data", help='Target directory')
parser.add_argument(
    '--download', dest='download', action='store_true', help='Enable download')
parser.add_argument(
    '--extract', dest='extract',  action='store_true', help='Enable extract')
parser.add_argument(
    '--convert', dest='convert',  action='store_true', help='Enable convert')
args = parser.parse_args()


# ========== ===========
# MD5SUM
# ========== ===========
def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# ========== ===========
# Download with wget
# ========== ===========
def download(args):
    name = 'TEDLIUM_release-3.tgz'
    url = 'http://www.openslr.org/resources/51/TEDLIUM_release-3.tgz'

    out = subprocess.call(
        'wget %s -O %s/%s' % (url, args.save_path, name), shell=True)
    if out != 0:
        raise ValueError(
            'Download failed %s. If download fails repeatedly'
            ', use alternate URL on the VoxCeleb website.' % url)


# ========== ===========
# Extract zip files
# ========== ===========
def extract(args):
    fname = '%s/TEDLIUM_release-3.tgz' % args.save_path
    print('Extracting %s' % fname)
    tar = tarfile.open(fname, "r:gz")
    tar.extractall(path=args.save_path)
    tar.close()


# ========== ===========
# Convert sph files
# ========== ===========
def convert(args):
    PAUSE_MATCH = re.compile(r'\([0-9]\)')
    NOTATION = re.compile(r'\{[A-Z]*\}')
    print('Converting .sph to wav')
    # splits = ['test']
    labels = []

    root = os.path.join(args.save_path, 'TEDLIUM_release-3', 'data')
    wav_dir = os.path.join(root, 'wav')
    os.makedirs(wav_dir, exist_ok=True)
    sph_files = sorted(
        list(glob.glob(os.path.join(root, 'sph/*.sph'))))
    with tqdm(sph_files, dynamic_ncols=True, desc="data") as pbar:
        for sph_file in pbar:
            sph = SPHFile(sph_file)
            stm_file = sph_file.replace('sph', 'stm')
            with open(stm_file, 'r') as f:
                for idx, line in enumerate(f.readlines()):
                    tokens = line.split(' ')
                    start, end = float(tokens[3]), float(tokens[4])
                    name = tokens[0]
                    text = line.split('male> ')[-1]
                    text = text.split('unknown> ')[-1]
                    text = text.split('NA> ')[-1]
                    text = text.replace('<sil>', '')
                    text = text.replace('<unk>', '')
                    text = text.split('('+name)[0]
                    text = PAUSE_MATCH.sub('', text)
                    text = NOTATION.sub('', text)
                    text = text.strip()
                    text = ' '.join(text.split())

                    wav_filename = '%s_%d.wav' % (name, idx)
                    assert ' ' not in wav_filename
                    sph.write_wav(
                        os.path.join(wav_dir, wav_filename),
                        start, end)
                    labels.append('%s %s' % (wav_filename, text))
        with open(os.path.join(wav_dir, 'labels.txt'), 'w') as f:
            f.write('\n'.join(labels))


# ========== ===========
# Main script
# ========== ===========
if __name__ == "__main__":
    if not os.path.exists(args.save_path):
        raise ValueError('Target directory does not exist.')

    if args.download:
        download(args)

    if args.extract:
        extract(args)

    if args.convert:
        convert(args)
