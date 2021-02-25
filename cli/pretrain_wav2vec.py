import torch
from rnnt.tokenizer import CharTokenizer, HuggingFaceTokenizer
from rnnt.transforms import build_transform, TrimAudio
from rnnt.args import FLAGS
from rnnt.dataset import Librispeech, seq_collate, MergedDataset
from torch.utils.data import DataLoader, Dataset
from modules.optimizer import AdamW
from tensorboardX import SummaryWriter
from rnnt.wav2vec import Wav2Vec, ConstrastiveCriterion
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

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
        batch_size=50, shuffle=True, num_workers=4,
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
        frontend_params = [(10, 5, 32)]+[(3, 2, 128)]*4 + [(2,2,128)] *3,
        front_bias=True,
        quantize_input=False,
        quantize_targets=True,
        input_size=128,
        enc_hidden_size=512, enc_layers=4, enc_dropout=0.1, enc_proj_size=512,
        num_negatives=100,
        feature_grad_mult=0.1,
        latent_temp=(1, 0.1, 0.999995),
    )
    model = model.cuda()

    constrast_learner = ConstrastiveCriterion(infonce=True, 
        loss_weights=[0.1, 1], 
        log_keys=["prob_perplexity", "code_perplexity", "temp"])

    global_step = 0
    optimizer = AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(0.9,0.98), eps=1e-06)

    total_epochs = 50
    total_iterations = len(dataloader) * total_epochs
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5000, 
            num_training_steps=total_iterations)

    tensorboard = SummaryWriter('logging/wav2vec_test1')

    eval_output = evaluate(model, val_dataloader)
    for key, value in eval_output.items():
        tensorboard.add_scalar('val/'+key, value, global_step)


    for e in tqdm(range(total_epochs), dynamic_ncols=True):
        for batch in dataloader:
            optimizer.zero_grad()
            raw_audio = batch[0]

            raw_audio = raw_audio.cuda()
            loss, sample_size, logging_output = constrast_learner(model, raw_audio, reduce=False)

            scale = 1.0
            if 'num_samples' in logging_output:
                scale = logging_output['num_samples']
            loss = loss / scale
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()
            lr_scheduler.step()

            if model.quantizer:
                model.quantizer.set_num_updates(global_step)
            if model.input_quantizer:
                model.input_quantizer.set_num_updates(global_step)

            for key, value in logging_output.items():
                tensorboard.add_scalar('train/'+key, value, global_step)
            global_step += 1
            # print(logging_output)

        if (e+1) % 2:
            eval_output = evaluate(model, val_dataloader)
            for key, value in eval_output.items():
                tensorboard.add_scalar('val/'+key, value, global_step)
            tensorboard.flush()

        torch.save(model.state_dict(), 'pretrained_test.pt')
