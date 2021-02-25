from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('name', 'rnn-t-v5', help='session name')
flags.DEFINE_enum('mode', 'train', ['train', 'resume', 'eval'], help='mode')
flags.DEFINE_integer('resume_step', None, help='model step')
# dataset
flags.DEFINE_string('LibriSpeech_train_100',
                    "../librispeech/LibriSpeech/train-clean-100",
                    help='LibriSpeech train')
flags.DEFINE_string('LibriSpeech_train_360',
                    "../librispeech/LibriSpeech/train-clean-360",
                    help='LibriSpeech train')
flags.DEFINE_string('LibriSpeech_train_500',
                    "../librispeech/LibriSpeech/train-other-500",
                    help='LibriSpeech train')
flags.DEFINE_string('LibriSpeech_test',
                    "../librispeech/LibriSpeech/test-clean",
                    help='LibriSpeech test')
flags.DEFINE_string('LibriSpeech_dev',
                    "../librispeech/LibriSpeech/dev-clean",
                    help='LibriSpeech dev')
flags.DEFINE_string('TEDLIUM_train',
                    "../speech_data/TEDLIUM/TEDLIUM_release1/train",
                    help='TEDLIUM 1 train')
flags.DEFINE_string('TEDLIUM_test',
                    "../speech_data/TEDLIUM/TEDLIUM_release1/test",
                    help='TEDLIUM 1 test')
flags.DEFINE_string('CommonVoice', "../speech_data/common_voice",
                    help='common voice')
flags.DEFINE_string('YT_bloomberg2', "../speech_data/common_voice",
                    help='common voice')
flags.DEFINE_string('YT_life', "../speech_data/common_voice",
                    help='common voice')

flags.DEFINE_integer('num_workers', 4, help='dataloader workers')
# learning
flags.DEFINE_bool('use_pretrained', default=False, help='Use pretrained enncoder')
flags.DEFINE_enum('optim', "adam", ['adam', 'sgd', 'sm3'], help='optimizer')
flags.DEFINE_float('lr', 1e-4, help='initial lr')
flags.DEFINE_bool('sched', True, help='lr reduce rate on plateau')
flags.DEFINE_integer('sched_patience', 1, help='lr reduce rate on plateau')
flags.DEFINE_float('sched_factor', 0.5, help='lr reduce rate on plateau')
flags.DEFINE_float('sched_min_lr', 1e-6, help='lr reduce rate on plateau')
flags.DEFINE_integer('warmup_step', 10000, help='linearly warmup lr')
flags.DEFINE_integer('epochs', 30, help='epoch')
flags.DEFINE_integer('batch_size', 8, help='batch size')
flags.DEFINE_integer('sub_batch_size', 8, help='accumulate batch size')
flags.DEFINE_integer('eval_batch_size', 4, help='evaluation batch size')
flags.DEFINE_float('gradclip', None, help='clip norm value')
# encoder
flags.DEFINE_string('enc_type', 'LSTM', help='encoder rnn type')
flags.DEFINE_integer('enc_hidden_size', 600, help='encoder hidden dimension')
flags.DEFINE_integer('enc_layers', 4, help='encoder layers')
flags.DEFINE_integer('enc_proj_size', 600, help='encoder layers')
flags.DEFINE_float('enc_dropout', 0, help='encoder dropout')
# decoder
flags.DEFINE_integer('dec_hidden_size', 150, help='decoder hidden dimension')
flags.DEFINE_integer('dec_layers', 2, help='decoder layers')
flags.DEFINE_integer('dec_proj_size', 150, help='encoder layers')
flags.DEFINE_float('dec_dropout', 0., help='decoder dropout')
# joint
flags.DEFINE_integer('joint_size', 512, help='Joint hidden dimension')
# tokenizer
flags.DEFINE_enum('tokenizer', 'char', ['char', 'bpe'], help='tokenizer')
flags.DEFINE_integer('bpe_size', 256, help='BPE vocabulary size')
flags.DEFINE_integer('vocab_embed_size', 16, help='vocabulary embedding size')
# data preprocess
flags.DEFINE_float('audio_max_length', 14, help='max length in seconds')
flags.DEFINE_enum('feature', 'mfcc', ['mfcc', 'melspec', 'logfbank'],
                  help='audio feature')
flags.DEFINE_integer('feature_size', 80, help='mel_bins')
flags.DEFINE_integer('n_fft', 400, help='spectrogram')
flags.DEFINE_integer('win_length', 400, help='spectrogram')
flags.DEFINE_integer('hop_length', 200, help='spectrogram')
flags.DEFINE_bool('delta', False, help='concat delta and detal of dealt')
flags.DEFINE_bool('cmvn', False, help='normalize spectrogram')
flags.DEFINE_integer('downsample', 3, help='downsample audio feature')
flags.DEFINE_integer('T_mask', 50, help='downsample audio feature')
flags.DEFINE_integer('T_num_mask', 2, help='downsample audio feature')
flags.DEFINE_integer('F_mask', 5, help='downsample audio feature')
flags.DEFINE_integer('F_num_mask', 1, help='downsample audio feature')
# apex
flags.DEFINE_bool('apex', default=True, help='fp16 training')
flags.DEFINE_string('opt_level', 'O1', help='use mix precision')
# parallel
flags.DEFINE_bool('multi_gpu', False, help='DataParallel')
# log
flags.DEFINE_integer('loss_step', 5, help='frequency to show loss in pbar')
flags.DEFINE_integer('save_step', 10000, help='frequency to save model')
flags.DEFINE_integer('eval_step', 10000, help='frequency to save model')
flags.DEFINE_integer('sample_size', 20, help='size of visualized examples')
