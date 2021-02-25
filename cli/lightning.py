import os, sys

import jiwer
import torch
import numpy as np
import torch.optim as optim
from absl import app
from apex import amp
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from warprnnt_pytorch import RNNTLoss
from rnnt.args import FLAGS
from rnnt.dataset import seq_collate, MergedDataset, Librispeech, CommonVoice, TEDLIUM, YoutubeCaption
from rnnt.models import Transducer
from rnnt.tokenizer import HuggingFaceTokenizer, CharTokenizer
from rnnt.transforms import build_transform
from rnnt.tokenizer import NUL, BOS, PAD
import pytorch_lightning as pl
from modules.optimizer import SM3, AdamW, Novograd
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

FLAGS(sys.argv)

class ParallelTraining(pl.LightningModule):
    def __init__(self):
        super(ParallelTraining, self).__init__()
        _, _, input_size = build_transform(
            feature_type=FLAGS.feature, feature_size=FLAGS.feature_size,
            n_fft=FLAGS.n_fft, win_length=FLAGS.win_length,
            hop_length=FLAGS.hop_length, delta=FLAGS.delta, cmvn=FLAGS.cmvn,
            downsample=FLAGS.downsample,
            T_mask=FLAGS.T_mask, T_num_mask=FLAGS.T_num_mask,
            F_mask=FLAGS.F_mask, F_num_mask=FLAGS.F_num_mask
        )
        self.log_path = None
        self.loss_fn = RNNTLoss(blank=NUL)
        
        if FLAGS.tokenizer == 'char':
            self.tokenizer = CharTokenizer(cache_dir=self.logdir)
        else:
            self.tokenizer = HuggingFaceTokenizer(
                cache_dir='BPE-2048', vocab_size=FLAGS.bpe_size)
        self.vocab_size = self.tokenizer.vocab_size
        print(FLAGS.enc_type)

        self.model = Transducer(
            vocab_embed_size=FLAGS.vocab_embed_size,
            vocab_size=self.vocab_size,
            input_size=input_size,
            enc_hidden_size=FLAGS.enc_hidden_size,
            enc_layers=FLAGS.enc_layers,
            enc_dropout=FLAGS.enc_dropout,
            enc_proj_size=FLAGS.enc_proj_size,
            dec_hidden_size=FLAGS.dec_hidden_size,
            dec_layers=FLAGS.dec_layers,
            dec_dropout=FLAGS.dec_dropout,
            dec_proj_size=FLAGS.dec_proj_size,
            joint_size=FLAGS.joint_size,
            module_type=FLAGS.enc_type,
            output_loss=False,
        )
        self.latest_alignment = None
        self.steps = 0
        self.epoch = 0
        self.best_wer = 1000

    def warmup_optimizer_step(self, steps):
        if steps < FLAGS.warmup_step:
            lr_scale = min(1., float(steps + 1) / FLAGS.warmup_step*1.0)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr_scale * FLAGS.lr
    
    def forward(self, batch):
        xs, ys, xlen, ylen = batch
        # xs, ys, xlen = xs.cuda(), ys, xlen.cuda()
        alignment = self.model(xs, ys, xlen, ylen)
        return alignment

    def training_step(self, batch, batch_nb):
        xs, ys, xlen, ylen = batch
        # xs, ys, xlen = xs.cuda(), ys, xlen.cuda()
        if xs.shape[1] != xlen.max():
            xs = xs[:, :xlen.max()]
            ys = ys[:, :ylen.max()]
        alignment = self.model(xs, ys, xlen, ylen)
        xlen = self.model.scale_length(alignment, xlen)
        loss = self.loss_fn(alignment, ys.int(), xlen, ylen)

        if batch_nb % 100 == 0:
            lr_val = 0
            for param_group in self.optimizer.param_groups:
                lr_val = param_group['lr']
            self.logger.experiment.add_scalar('lr', lr_val, self.steps)

        self.steps += 1

        if self.steps < FLAGS.warmup_step:
            self.warmup_optimizer_step(self.steps)

        return {'loss': loss, 'log': {
            'loss': loss.item()
        }}

    def validation_step(self, batch, batch_nb):
        xs, ys, xlen, ylen = batch
        y, nll = self.model.greedy_decode(xs, xlen)

        hypothesis = self.tokenizer.decode_plus(y)
        ground_truth = self.tokenizer.decode_plus(ys.cpu().numpy())
        measures = jiwer.compute_measures(ground_truth, hypothesis)

        return {'val_loss': nll.mean().item(), 'wer': measures['wer'], 'ground_truth': ground_truth[0], 'hypothesis': hypothesis[0]}

    def validation_end(self, outputs):
        # OPTIONAL
        self.logger.experiment.add_text('test', 'This is test', 0)

        avg_wer = np.mean([x['wer'] for x in outputs])
        ppl = np.mean([x['val_loss'] for x in outputs])
        self.logger.experiment.add_scalar('val/WER', avg_wer, self.steps)
        self.logger.experiment.add_scalar('val/perplexity', ppl, self.steps)

        hypothesis, ground_truth = '', ''
        for idx in range(min(5, len(outputs))):
            hypothesis += outputs[idx]['hypothesis']+'\n\n'
            ground_truth += outputs[idx]['ground_truth'] + '\n\n'

        self.logger.experiment.add_text('generated', hypothesis, self.steps)
        self.logger.experiment.add_text('grouth_truth', ground_truth, self.steps)
        if self.latest_alignment != None:
            alignment = self.latest_alignment
            idx = random.randint(0, alignment.size(0) - 1)
            alignment = torch.softmax(alignment[idx], dim=-1)
            alignment[:, :, 0] = 0 # ignore blank token
            alignment = alignment.mean(dim=-1)

            self.logger.experiment.add_image(
                    "alignment",
                    plot_alignment_to_numpy(alignment.data.numpy().T),
                    self.steps, dataformats='HWC')
        self.logger.experiment.flush()

        if self.best_wer > avg_wer and self.epoch > 0:
            print('best checkpoint found!')
            # checkpoint = {
            #     'model': self.model.state_dict(),
            #     'optimizer': self.optimizer.state_dict(),
            #     'epoch': self.epoch
            # }
            # if FLAGS.apex:
            #     checkpoint['amp'] = amp.state_dict()
            # torch.save(checkpoint, os.path.join(self.log_path, str(self.epoch)+'amp_checkpoint.pt'))
            self.trainer.save_checkpoint(os.path.join(self.log_path, str(self.epoch)+'amp_checkpoint.pt'))

            self.best_wer = avg_wer


        self.plateau_scheduler.step(avg_wer)
        self.epoch += 1

        return {'val/WER': torch.tensor(avg_wer),
            'wer': torch.tensor(avg_wer),
            'val/perplexity': torch.tensor(ppl) }
    
    def validation_epoch_end(self, outputs):
        avg_wer = np.mean([x['wer'] for x in outputs])
        ppl = np.mean([x['val_loss'] for x in outputs])

        hypothesis, ground_truth = '', ''
        for idx in range(5):
            hypothesis += outputs[idx]['hypothesis']+'\n\n'
            ground_truth += outputs[idx]['ground_truth'] + '\n\n'

        writer.add_text('generated', hypothesis, self.steps)
        writer.add_text('grouth_truth', ground_truth, self.steps)

        if self.latest_alignment != None:
            alignment = self.latest_alignment
            idx = random.randint(0, alignment.size(0) - 1)
            alignment = torch.softmax(alignment[idx], dim=-1)
            alignment[:, :, 0] = 0 # ignore blank token
            alignment = alignment.mean(dim=-1)

            writer.add_image(
                    "alignment",
                    plot_alignment_to_numpy(alignment.data.numpy().T),
                    self.steps, dataformats='HWC')

        self.logger.experiment.add_scalar('val/WER', avg_wer, self.steps)
        self.logger.experiment.add_scalar('val/perplexity', ppl, self.steps)
        self.logger.experiment.flush()

        self.plateau_scheduler.step(avg_wer)

        self.epoch += 1
        return {'val/WER': torch.tensor(avg_wer),
         'val/perplexity': torch.tensor(ppl) }

    def configure_optimizers(self):
        if FLAGS.optim == 'adam':
            self.optimizer = AdamW(
                self.model.parameters(), lr=FLAGS.lr, weight_decay=1e-5)
        elif FLAGS.optim == 'sm3':
            self.optimizer = SM3(
                self.model.parameters(), lr=FLAGS.lr, momentum=0.0)
        else:
            self.optimizer = Novograd(
                self.model.parameters(), lr=FLAGS.lr, weight_decay=1e-3)
        scheduler = []
        if FLAGS.sched:
            self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=FLAGS.sched_patience,
                factor=FLAGS.sched_factor, min_lr=FLAGS.sched_min_lr,
                verbose=1)
            scheduler= [self.plateau_scheduler]

        self.warmup_optimizer_step(0)
        return [self.optimizer]

    @pl.data_loader
    def train_dataloader(self):
        transform_train, _, _ = build_transform(
            feature_type=FLAGS.feature, feature_size=FLAGS.feature_size,
            n_fft=FLAGS.n_fft, win_length=FLAGS.win_length,
            hop_length=FLAGS.hop_length, delta=FLAGS.delta, cmvn=FLAGS.cmvn,
            downsample=FLAGS.downsample,
            T_mask=FLAGS.T_mask, T_num_mask=FLAGS.T_num_mask,
            F_mask=FLAGS.F_mask, F_num_mask=FLAGS.F_num_mask
        )

        dataloader = DataLoader(
            dataset=MergedDataset([
                Librispeech(
                    root=FLAGS.LibriSpeech_train_500,
                    tokenizer=self.tokenizer,
                    transform=transform_train,
                    audio_max_length=FLAGS.audio_max_length),
                Librispeech(
                    root=FLAGS.LibriSpeech_train_360,
                    tokenizer=self.tokenizer,
                    transform=transform_train,
                    audio_max_length=FLAGS.audio_max_length),
                # Librispeech(
                #     root=FLAGS.LibriSpeech_train_100,
                #     tokenizer=self.tokenizer,
                #     transform=transform_train,
                #     audio_max_length=FLAGS.audio_max_length),
                TEDLIUM(
                    root=FLAGS.TEDLIUM_train,
                    tokenizer=self.tokenizer,
                    transform=transform_train,
                    audio_max_length=FLAGS.audio_max_length),
                CommonVoice(
                    root=FLAGS.CommonVoice, labels='train.tsv',
                    tokenizer=self.tokenizer,
                    transform=transform_train,
                    audio_max_length=FLAGS.audio_max_length,
                    audio_min_length=1),
                YoutubeCaption(
                    root='../speech_data/youtube-speech-text/', labels='bloomberg2_meta.csv',
                    tokenizer=self.tokenizer,
                    transform=transform_train,
                    audio_max_length=FLAGS.audio_max_length,
                    audio_min_length=1),
                YoutubeCaption(
                    root='../speech_data/youtube-speech-text/', labels='life_meta.csv',
                    tokenizer=self.tokenizer,
                    transform=transform_train,
                    audio_max_length=FLAGS.audio_max_length,
                    audio_min_length=1),                    
                YoutubeCaption(
                    root='../speech_data/youtube-speech-text/', labels='news_meta.csv',
                    tokenizer=self.tokenizer,
                    transform=transform_train,
                    audio_max_length=FLAGS.audio_max_length,
                    audio_min_length=1),
                YoutubeCaption(
                    root='../speech_data/youtube-speech-text/', labels='english2_meta.csv',
                    tokenizer=self.tokenizer,
                    transform=transform_train,
                    audio_max_length=FLAGS.audio_max_length,
                    audio_min_length=1),
            ]),
            batch_size=FLAGS.sub_batch_size, shuffle=True,
            num_workers=FLAGS.num_workers, collate_fn=seq_collate,
            drop_last=True)
        return dataloader

    @pl.data_loader
    def val_dataloader(self):
        _, transform_test, _ = build_transform(
            feature_type=FLAGS.feature, feature_size=FLAGS.feature_size,
            n_fft=FLAGS.n_fft, win_length=FLAGS.win_length,
            hop_length=FLAGS.hop_length, delta=FLAGS.delta, cmvn=FLAGS.cmvn,
            downsample=FLAGS.downsample,
            T_mask=FLAGS.T_mask, T_num_mask=FLAGS.T_num_mask,
            F_mask=FLAGS.F_mask, F_num_mask=FLAGS.F_num_mask
        )

        val_dataloader = DataLoader(
            dataset=MergedDataset([
                Librispeech(
                    root=FLAGS.LibriSpeech_test,
                    tokenizer=self.tokenizer,
                    transform=transform_test,
                    reverse_sorted_by_length=True)]),
            batch_size=FLAGS.eval_batch_size, shuffle=False,
            num_workers=FLAGS.num_workers, collate_fn=seq_collate)
        return val_dataloader




if __name__ == "__main__":
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    import pickle
    model = ParallelTraining()
    # with open('test.pt', 'wb') as f:
    #     pickle.dump(model, f)
    gpus = [0,1, 2, 3]
    params = {
        'gpus': gpus,
        'distributed_backend': 'ddp',
        'gradient_clip_val': 10,
        'val_check_interval': 0.25,
        'accumulate_grad_batches': FLAGS.batch_size // (FLAGS.sub_batch_size*len(gpus))
    }
    if  FLAGS.apex:
        print('use apex')
        params['amp_level'] = FLAGS.opt_level
        params['precision'] = 16
        params['min_loss_scale'] = 1.0

    from datetime import datetime
    cur_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_name = '{}-{}'.format('rnnt-m', FLAGS.tokenizer)
    log_path = 'logs/{}'.format(log_name)
    os.makedirs(log_path, exist_ok=True)

    model.log_path = log_path
    logger = pl.loggers.tensorboard.TensorBoardLogger('logs', name='rnnt-m')
    params['logger'] = logger

    checkpoint_callback = ModelCheckpoint(
        filepath=log_path,
        save_top_k=True,
        verbose=True,
        monitor='val/perplexity',
        mode='min',
        prefix=''
    )
    params['checkpoint_callback'] = checkpoint_callback
    print(params)
    # params['resume_from_checkpoint'] = '/home/theblackcat/rnn_transducer/logs/rnnt-bpe/8amp_checkpoint.pt'
    trainer = Trainer(**params)
    model.trainer = trainer
    trainer.fit(model)

