from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('name', 'wav2vec_test', help='session name')

flags.DEFINE_float('lr', 1e-4, help='initial lr')
flags.DEFINE_integer('warmup_step', 10000, help='linearly warmup lr')
flags.DEFINE_integer('epochs', 50, help='epoch')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('sub_batch_size', 24, help='accumulate batch size')

flags.DEFINE_float('gradclip', None, help='clip norm value')

flags.DEFINE_float('prob_perplex', 0.1, help='prob_perplexity weight')
flags.DEFINE_float('code_perplex', 1, help='code_perplexity weight')

flags.DEFINE_float('init_temp', 1, help='initial temperature value')
flags.DEFINE_float('min_temp', 0.1, help='minimal temperature value')
flags.DEFINE_float('temp_decay', 0.999995, help='minimal temperature value')
flags.DEFINE_integer('num_workers', 8, help='dataloader workers')
flags.DEFINE_integer('eval_iteration', 1000, help='evaluate every iterations')
flags.DEFINE_float('feature_grad_mult', 0.1, help='feature gradient multiplication weight')


flags.DEFINE_float('beta1', 0.9, help='adam beta 0')
flags.DEFINE_float('beta2', 0.998, help='adam beta - second momentum')
flags.DEFINE_float('weight_decay', 0.01, help='weight decay')


flags.DEFINE_integer('num_negatives', 100, help='num negatives sample for constrastive learning')
flags.DEFINE_integer('enc_proj_size', 512, help='encoder projection size')
flags.DEFINE_integer('enc_layers', 4, help='number of encoding layers')
flags.DEFINE_integer('enc_hidden_size', 512, help='number of encoder hidden size')
flags.DEFINE_float('enc_dropout', 0.1, help='encoder dropout')


flags.DEFINE_bool('multi_gpu', False, help='DataParallel')