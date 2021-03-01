# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Zhi Rui Tam and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F
from torch import nn

from rnnt.tokenizer import NUL, BOS, PAD
from rnnt.models import Encoder, ResLayerNormGRU, ResLayerNormLSTM, FrontEnd
from modules.kmeans_vector_quantizer import KmeansVectorQuantizer
from modules.softmax_vector_quantizer import GumbelVectorQuantizer
from rnnt.data_utils import compute_mask_indices, buffered_arange




class Wav2Vec(nn.Module):

    def __init__(self,
                # frontend module parameters
                 frontend_params = [(10, 5, 16)]+[ (8, 4, 32) ]+[(4,2,128)]*3,
                 front_bias=False,
                # encoder parameters
                 input_size=768,
                 enc_hidden_size=768, enc_layers=7, enc_dropout=0.1, enc_proj_size=512,
                 blank=NUL, module_type='LSTM', output_loss=True,
                # quantization learning
                 quantize_input=False,
                 quantize_targets=False,
                 same_quantizer=False,
                 mask_prob=0.15,
                 mask_length=10,
                 mask_selection='static', # "static", "uniform", "normal", "poisson"
                 mask_other=0.0,
                 mask_channel_prob=0.0,
                 mask_channel_selection='static', # "static", "uniform"
                 mask_channel_other=0,
                 mask_channel_min_space=1,
                 no_mask_channel_overlap=False,
                 no_mask_overlap=False,
                 mask_min_space=1,
                 dropout_input=0.0,
                 dropout_features=0.0,
                 num_negatives=100,
                 negatives_from_everywhere=False,
                 cross_sample_negatives=0,
                 codebook_negatives=0,
                 final_dim=0,
                 latent_groups=2,
                 latent_dim=0,
                 target_glu=False,
                 latent_vars=320,
                 feature_grad_mult=1.0,
                 logit_temp=0.1,
                 latent_temp=(2, 0.5, 0.999995) # can be tuple of 3 values (start, end, decay)
                ):

        super().__init__()
        self.blank = blank
        self.quantize_input = quantize_input

        # Encoder
        if module_type not in ['GRU', 'LSTM']:
            raise ValueError('Unsupported module type')
        if module_type == 'GRU':
            module = ResLayerNormGRU
        else:
            module = ResLayerNormLSTM

        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=enc_hidden_size,
            num_layers=enc_layers,
            dropout=enc_dropout,
            time_reductions=[], # no time reduction
            proj_size=enc_proj_size,
            module=module )
        self.frontend = FrontEnd(frontend_params, bias=front_bias
        )

        self.encoder_embed_dim = input_size
        self.embed  = frontend_params[-1][-1]

        self.post_extract_proj = (
            nn.Linear(self.embed, input_size)
            if self.embed != input_size and not quantize_input
            else None
        )
        self.layer_norm = nn.LayerNorm(self.embed)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(self.encoder_embed_dim).uniform_()
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, self.encoder_embed_dim)
            if self.embed != self.encoder_embed_dim and not quantize_input
            else None
        )
        self.dropout_input = nn.Dropout(dropout_input)
        self.dropout_features = nn.Dropout(dropout_features)

        self.mask_prob = mask_prob
        self.mask_selection = mask_selection
        self.mask_channel_prob = mask_channel_prob
        self.mask_other = mask_other
        self.mask_length = mask_length
        self.no_mask_overlap = no_mask_overlap
        self.mask_min_space = mask_min_space

        self.quantizer = None
        self.input_quantizer = None
        self.n_negatives = num_negatives
        self.cross_sample_negatives = cross_sample_negatives
        self.codebook_negatives = codebook_negatives
        self.negatives_from_everywhere = negatives_from_everywhere

        self.logit_temp = logit_temp

        final_dim = final_dim if final_dim > 0 else self.encoder_embed_dim
        if quantize_targets:
            vq_dim = latent_dim if latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=latent_vars,
                temp=latent_temp,
                groups=latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
            )
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            self.project_q = nn.Linear(self.embed, final_dim)

        if quantize_input:
            if same_quantizer and self.quantizer is not None:
                vq_dim = final_dim
                self.input_quantizer = self.quantizer
            else:
                vq_dim = latent_dim if latent_dim > 0 else self.encoder_embed_dim
                self.input_quantizer = GumbelVectorQuantizer(
                    dim=self.embed,
                    num_vars=latent_vars,
                    temp=latent_temp,
                    groups=latent_groups,
                    combine_groups=False,
                    vq_dim=vq_dim,
                    time_first=True,
                )
            self.project_inp = nn.Linear(vq_dim, self.encoder_embed_dim)

        self.target_glu = None
        if target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.final_proj = nn.Linear(enc_proj_size, final_dim)


    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def sample_negatives(self, y, num):

        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        cross_high = tsz * bsz
        high = tsz
        with torch.no_grad():
            assert high > 1, f"{bsz,tsz,fsz}"

            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * num)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs


    def forward(self, source, padding_mask=None, mask=True, features_only=False):
        features = self.frontend(source)

        unmasked_features = features.clone()

        features_pen = features.clone().float().pow(2).mean()

        if padding_mask is not None:
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[(torch.arange(padding_mask.shape[0], device=padding_mask.device), output_lengths - 1)] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.project_inp(features)

        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask)
            if mask_indices is not None:
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        x, _ = self.encoder(x)

        if features_only:
            return {"x": x, "padding_mask": padding_mask}

        if self.quantizer:
            q = self.quantizer(y, produce_targets=not self.training )
            y_i = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            y = self.project_q(y_i)

            if self.negatives_from_everywhere:
                neg_cands, *_ = self.quantizer(unmasked_features, produce_targets=False)
                negs, _ = self.sample_negatives(neg_cands, y.size(1))
                negs = self.project_q(negs)

            else:
                negs, _ = self.sample_negatives(y, y.size(1))

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(unmasked_features, y.size(1))
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(y, y.size(1))

        x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        x = self.final_proj(x)
        x = self.compute_preds(x, y, negs)

        result = {"x": x, "padding_mask": padding_mask, "features_pen": features_pen}

        if not self.training:
            result['targets'] = q['targets']

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp

        return result


    def compute_preds(self, x, y, negatives):

        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)

        targets = torch.cat([y, negatives], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)

        logits /= self.logit_temp

        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")

        return logits

    def get_logits(self, net_output):
        logits = net_output["x"]
        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

    def get_targets(self, sample, net_output, expand_steps=True):
        x = net_output["x"]
        return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)

    def get_extra_losses(self, net_output):
        ''' Penality losses
        '''
        pen = []

        if "prob_perplexity" in net_output:
            pen.append(
                (net_output["num_vars"] - net_output["prob_perplexity"])
                / net_output["num_vars"]
            )

        if "features_pen" in net_output:
            pen.append(net_output["features_pen"])

        return pen


class ConstrastiveCriterion(nn.Module):
    # https://github.com/pytorch/fairseq/blob/0f93bd1a7d451944b77804aaf25e40696510411b/fairseq/criterions/wav2vec_criterion.py


    def __init__(self, infonce=False, loss_weights=None, log_keys=None):
        super().__init__()

        self.infonce = infonce
        self.loss_weights = loss_weights
        self.log_keys = [] if log_keys is None else log_keys

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        results = model(sample)

        # x => labels x B x T, labels[0] = pos, labels[1:] = neg
        logits = model.get_logits(results).float()
        target = model.get_targets(None, results)

        weights = None
        if hasattr(model, "get_target_weights") and not self.infonce:
            weights = model.get_target_weights(target, results)
            if torch.is_tensor(weights):
                weights = weights.float()

        losses = []
        reduce = True
        if self.infonce:
            loss = F.cross_entropy(
                logits,
                target,
                reduction="sum" if reduce else "none",
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits,
                target.float(),
                weights,
                reduction="sum" if reduce else "none",
            )

        sample_size = target.numel() if self.infonce else target.long().sum().item()
        losses.append(loss.detach().clone())

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses = model.get_extra_losses(results)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(
                self.loss_weights
            ), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    losses.append(p)

        logging_output = {
            "loss": loss.item() if reduce else loss,
            "ntokens": sample_size,
            # "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        for lk in self.log_keys:
            # Only store "logits" and "target" for computing MAP and MAUC
            # during validation
            if lk == "logits":
                if not self.training:
                    logging_output["logits"] = logits.cpu().numpy()
            elif lk == "target":
                if not self.training:
                    logging_output["target"] = target.cpu().numpy()
            elif lk in results:
                logging_output[lk] = float(results[lk])

        if len(losses) > 1:
            for i, l in enumerate(losses):
                logging_output[f"loss_{i}"] = l.item()

        if self.infonce:
            with torch.no_grad():
                if logits.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    assert logits.dim() > 1, logits.shape
                    max = logits.argmax(-1) == 0
                    min = logits.argmin(-1) == 0
                    both = max & min
                    corr = max.long().sum().item() - both.long().sum().item()
                    count = max.numel()

                logging_output["correct"] = corr
                logging_output["count"] = count

        return loss, sample_size, logging_output


