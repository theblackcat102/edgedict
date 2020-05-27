import argparse

import os
import subprocess
import glob
from tqdm import tqdm


parser = argparse.ArgumentParser(description="VoxCeleb downloader")
parser.add_argument('--save_path', 	type=str, default="data",
                    help='Target directory')
args = parser.parse_args()

'''
/mnt/ws/common_voice/clips/common_voice_en_105953.mp3.
'''


def convert(args):
    files = glob.glob('%s/clips/*.mp3' % args.save_path)
    files.sort()

    print('Converting files from MP3 to WAV')
    for fname in tqdm(files):
        outfile = fname.replace('.mp3', '.wav')
        if not os.path.exists(outfile):
            out = subprocess.call(
                'ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s '
                '>/dev/null 2>/dev/null' % (fname, outfile), shell=True)
            if out != 0:
                print('Conversion failed %s.' % fname)


if __name__ == '__main__':
    convert(args)
