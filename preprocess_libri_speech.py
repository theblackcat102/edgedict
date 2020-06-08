import argparse
import os
import subprocess
import hashlib
import glob
import tarfile

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
    files = [
        # (
        #     'dev-clean.tar.gz',
        #     'http://www.openslr.org/resources/12/dev-clean.tar.gz',
        #     '42e2234ba48799c1f50f24a7926300a1'
        # ),
        # (
        #     'dev-other.tar.gz',
        #     'http://www.openslr.org/resources/12/dev-other.tar.gz',
        #     'c8d0bcc9cca99d4f8b62fcc847357931'
        # ),
        # (
        #     'test-clean.tar.gz',
        #     'http://www.openslr.org/resources/12/test-clean.tar.gz',
        #     '32fa31d27d2e1cad72775fee3f4849a9'
        # ),
        # (
        #     'test-other.tar.gz',
        #     'http://www.openslr.org/resources/12/test-other.tar.gz',
        #     'fb5a50374b501bb3bac4815ee91d3135'
        # ),
        # (
        #     'train-clean-100.tar.gz',
        #     'http://www.openslr.org/resources/12/train-clean-100.tar.gz',
        #     '2a93770f6d5c6c964bc36631d331a522',
        # ),
        # (
        #     'train-clean-360.tar.gz',
        #     'http://www.openslr.org/resources/12/train-clean-360.tar.gz',
        #     'c0e676e450a7ff2f54aeade5171606fa',
        # ),
        (
            'train-other-500.tar.gz',
            'http://www.openslr.org/resources/12/train-other-500.tar.gz',
            'd1a0fd59409feb2c614ce4d30c387708',
        )
    ]
    for name, url, md5gt in files:
        # Download files
        out = subprocess.call(
            'wget %s -O %s/%s' % (url, args.save_path, name), shell=True)
        if out != 0:
            raise ValueError(
                'Download failed %s. If download fails repeatedly'
                ', use alternate URL on the VoxCeleb website.' % url)

        # Check MD5
        md5ck = md5('%s/%s' % (args.save_path, name))
        if md5ck == md5gt:
            print('Checksum successful %s.' % name)
        else:
            raise ValueError('Checksum failed %s.' % name)


# ========== ===========
# Extract zip files
# ========== ===========
def extract(args):
    files = glob.glob('%s/*.tar.gz' % args.save_path)

    for fname in files:
        print('Extracting %s' % fname)
        tar = tarfile.open(fname, "r:gz")
        tar.extractall(path=args.save_path)
        tar.close()


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
