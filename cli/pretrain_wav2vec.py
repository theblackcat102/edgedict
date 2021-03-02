import torch
import sys, os
from rnnt.tokenizer import CharTokenizer, HuggingFaceTokenizer
from rnnt.transforms import build_transform, TrimAudio
from rnnt.dataset import Librispeech, seq_collate, MergedDataset, YoutubeCaption
from torch.utils.data import DataLoader, Dataset
from modules.optimizer import AdamW
from tensorboardX import SummaryWriter
from rnnt.wav2vec import Wav2Vec, ConstrastiveCriterion
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from rnnt.pretrain_args import FLAGS

FLAGS(sys.argv)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        learning_rate = max(0.0, 1. - (float(current_step) / float(num_training_steps)))
        learning_rate *= min(1.0, float(current_step) / float(num_warmup_steps))
        return learning_rate

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_params_without_weight_decay_ln(named_params, weight_decay):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in named_params if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    return optimizer_grouped_parameters


def evaluate(model, dataloader):
    model.eval()
    logging_outputs = {}
    with torch.no_grad():
        for batch in dataloader:
            raw_audio = batch[0]

            raw_audio = raw_audio.cuda()
            loss, sample_size, logging_output = constrast_learner(model, raw_audio, reduce=False)
            for key, value in logging_output.items():
                if key not in logging_outputs:
                    logging_outputs[key] = []
                if FLAGS.multi_gpu and isinstance(value, torch.Tensor):
                    value = value.mean()

                logging_outputs[key].append(value)

    model.train()
    return {  key: np.mean(scores) for key, scores in logging_outputs.items() }


if __name__ == '__main__':
    # tokenizer is not needed in this stage
    tokenizer = HuggingFaceTokenizer(
            cache_dir='BPE-2048', vocab_size=2048)

    transform = torch.nn.Sequential(
        TrimAudio(sampling_rate=16000, max_audio_length=15)
    )

    dataloader = DataLoader(
        dataset=MergedDataset([
            YoutubeCaption(
                '../yt_speech/', labels='news_dummy.csv',
                tokenizer=tokenizer,
                transform=transform,
                audio_max_length=14,
            ),
            YoutubeCaption(
                '../yt_speech/', labels='life_dummy.csv',
                tokenizer=tokenizer,
                transform=transform,
                audio_max_length=14,
            ),
            Librispeech(
                '../librispeech/LibriSpeech/dev-clean',
                tokenizer=tokenizer,
                transform=transform),
            Librispeech(
                '../librispeech/LibriSpeech/dev-other',
                tokenizer=tokenizer,
                transform=transform),
            Librispeech(
                '../librispeech/LibriSpeech/test-other',
                tokenizer=tokenizer,
                transform=transform),
            Librispeech(
                '../librispeech/LibriSpeech/train-clean-360',
                tokenizer=tokenizer,
                transform=transform),
            Librispeech(
                '../librispeech/LibriSpeech/train-clean-100',
                tokenizer=tokenizer,
                transform=transform),
        ]),
        batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers,
        collate_fn=seq_collate
    )

    val_dataloader =  DataLoader(
                dataset=Librispeech(
                '../librispeech/LibriSpeech/test-clean',
                tokenizer=tokenizer,
                transform=transform),
            batch_size=50, shuffle=True, num_workers=4,
            collate_fn=seq_collate
        )

    model = Wav2Vec(
        frontend_params = [(10, 5, 32)]+[(3, 2, 128)]*4 + [(2, 2, 128)] *3,
        front_bias=False,
        quantize_input=False,
        quantize_targets=True,
        input_size=128,
        enc_hidden_size=FLAGS.enc_hidden_size,
        enc_layers=FLAGS.enc_layers,
        enc_dropout=FLAGS.enc_dropout,
        enc_proj_size=FLAGS.enc_proj_size,
        num_negatives=FLAGS.num_negatives,
        feature_grad_mult=FLAGS.feature_grad_mult,
        latent_temp=(FLAGS.init_temp, FLAGS.min_temp, FLAGS.temp_decay),
    )

    model = model.cuda()

    constrast_learner = ConstrastiveCriterion(infonce=True,
        loss_weights=[FLAGS.prob_perplex, FLAGS.code_perplex],
        log_keys=["prob_perplexity", "code_perplexity", "temp"])

    global_step = 0
    optimizer = AdamW(
        get_params_without_weight_decay_ln(model.named_parameters(), FLAGS.weight_decay),
        lr=FLAGS.lr, betas=(FLAGS.beta1, FLAGS.beta2), eps=1e-08)

    total_epochs = FLAGS.epochs
    total_iterations = len(dataloader) * total_epochs
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=FLAGS.warmup_step,
            num_training_steps=total_iterations)

    tensorboard = SummaryWriter('logging/'+FLAGS.name)
    tensorboard.add_text(
        'flagfile', FLAGS.flags_into_string().replace('\n', '\n\n'))
    FLAGS.append_flags_into_file(os.path.join('logging/'+FLAGS.name, 'flagfile.txt'))

    eval_output = evaluate(model, val_dataloader)
    for key, value in eval_output.items():
        tensorboard.add_scalar('val/'+key, value, global_step)

    if FLAGS.multi_gpu:
        model = torch.nn.DataParallel(model)


    start_idxs = list(range(0, FLAGS.batch_size, FLAGS.sub_batch_size))
    best_correct = -1
    with tqdm(total=int(len(dataloader)*total_epochs), dynamic_ncols=True) as pbar:
        for e in range(total_epochs):

            for batch in dataloader:
                optimizer.zero_grad()

                start_idxs = range(0, FLAGS.batch_size, FLAGS.sub_batch_size)
                losses = []
                logging_outputs = {}

                for sub_batch_idx, start_idx in enumerate(start_idxs):
                    sub_slice = slice(start_idx, start_idx + FLAGS.sub_batch_size)
                    raw_audio = batch[0]
                    raw_audios = raw_audio[sub_slice]

                    if len(raw_audios) > 0:
                        raw_audios = raw_audios.cuda()
                        loss, sample_size, logging_output = constrast_learner(model, raw_audios, reduce=False)
                        if FLAGS.multi_gpu:
                            loss = loss.mean() / len(start_idxs)
                        else:
                            loss = loss / len(start_idxs)

                        loss.backward()
                        losses.append(loss)

                loss = torch.stack(losses).mean()

                if FLAGS.gradclip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), FLAGS.gradclip)

                optimizer.step()
                lr_scheduler.step()

                if model.quantizer:
                    model.quantizer.set_num_updates(global_step)
                if model.input_quantizer:
                    model.input_quantizer.set_num_updates(global_step)

                for key, value in logging_output.items():
                    if FLAGS.multi_gpu and isinstance(value, torch.Tensor):
                        value = value.mean()
                    tensorboard.add_scalar('train/'+key, value, global_step)

                global_step += 1
                pbar.update(1)
                pbar.set_description('Epoch %d, loss: %.4f' % (e, loss.item()))

                if (global_step) % FLAGS.eval_iteration == 0:
                    eval_output = evaluate(model, val_dataloader)
                    for key, value in eval_output.items():
                        tensorboard.add_scalar('val/'+key, value, global_step)
                    tensorboard.flush()
                    if eval_output['correct'] > best_correct:
                        if FLAGS.multi_gpu:
                            torch.save(model.module.state_dict(), 'pretrained_test.pt')
                        else:
                            torch.save(model.state_dict(), 'pretrained_test.pt')
                        best_correct = eval_output['correct']