import argparse

import os
import subprocess
import glob
from tqdm import tqdm
import argparse
import os
import glob
import multiprocessing
import json
import pandas as pd


from rnnt.preprocessing_utils import parallel_preprocess

parser = argparse.ArgumentParser(description="VoxCeleb downloader")
parser.add_argument('--clip_path', 	type=str, default="data",
                    help='mp3 directory')
parser.add_argument('--save_path', 	type=str, default="data",
                    help='Target directory')
parser.add_argument('--target_sr', type=int, default=16000,
                    help='Target sample rate. '
                         'defaults to the input sample rate')
parser.add_argument('--parallel', type=int, default=multiprocessing.cpu_count(),
                    help='Number of threads to use when processing audio files')
args = parser.parse_args()

'''
/mnt/ws/common_voice/clips/common_voice_en_105953.mp3.
'''


def convert(args):
    datasets = glob.glob('%s/*.mp3' % args.clip_path)
    datasets.sort()
    dataset = parallel_preprocess(datasets, '', args.save_path, args.target_sr, None, True, args.parallel)
    print("[%s] Generating json..." % args.output_json)
    df = pd.DataFrame(dataset, dtype=object)

    # Save json with python. df.to_json() produces back slashed in file paths
    dataset = df.to_dict(orient='records')
    with open(args.output_json, 'w') as fp:
        json.dump(dataset, fp, indent=2)


if __name__ == '__main__':
    convert(args)
