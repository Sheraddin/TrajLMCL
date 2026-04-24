import os
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BertTokenizerFast
from peft import LoraModel, LoraConfig
from einops import repeat

from .base import Encoder
from .layers import PatternSemanticProjector, TrajEmbedding, TrajConvEmbedding


def get_batch_mask(B, L, valid_len):
    mask = repeat(torch.arange(end=L, device=valid_len.device),
                  'L -> B L', B=B) < repeat(valid_len, 'B -> B L', L=L)  # (B, L)
    return mask


def _mean_pooling(model_output, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
    return torch.sum(model_output * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def concatenate_sequences(original_tensor, additional_tensor, original_valid_length, additional_valid_length=None):
    """
    Concatenate additional sequences to the original sequences, moving padding to the end.
    """
    B, L, feature_dim = original_tensor.size()
    additional_length = additional_tensor.size(1)
    if additional_valid_length is None:
        additional_valid_length = torch.full((B,), additional_length, dtype=torch.long).to(original_tensor.device)
    if isinstance(original_valid_length, int):
        original_valid_length = torch.full((B,), L, dtype=torch.long).to(original_tensor.device)

    # Create masks
    res_length = (additional_valid_length + original_valid_length).max()
    additional_batch_mask = get_batch_mask(B, additional_length, additional_valid_length)
    valid_batch_mask = get_batch_mask(B, res_length, original_valid_length)
    add_additional_batch_mask = get_batch_mask(B, res_length, original_valid_length + additional_valid_length)

    # Initialize the result tensor
    res_tensor = torch.zeros(B, res_length, feature_dim).to(original_tensor.device)
    res_tensor[:, :L] = original_tensor

    # Use masks to concatenate additional sequences and move padding
    res_tensor[(add_additional_batch_mask.long() - valid_batch_mask.long()) == 1] = additional_tensor[additional_batch_mask]

    return res_tensor, original_valid_length + additional_valid_length


def get_model_path(model_class):
    return os.path.join(os.getcwd(), 'params/{}'.format(model_class))


def get_encoder(model_path, model_class, tokenizer=None,
                lora=True, lora_alpha=32, lora_dim=8):
    if model_class == 'gpt2':
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=lora_dim,
            lora_alpha=lora_alpha,
            target_modules=["c_attn"],
            lora_dropout=0.02,
        )
        model = AutoModelForCausalLM.from_pretrained(model_path)
        if tokenizer is not None:
            model.resize_token_embeddings(len(tokenizer))
        emb_size = model.config.n_embd
        hidden_size = model.config.n_embd
        word_embedding = model.get_input_embeddings()

    else:
        raise NotImplementedError("model_class should be one of ['gpt2']")

    if lora:
        return LoraModel(model, lora_config, model_class), emb_size, hidden_size, word_embedding
    else:
        return model, emb_size, hidden_size, word_embedding


def get_tokenizer(model_path, model_class):
    return AutoTokenizer.from_pretrained(model_path)


class LET(Encoder):
    def __init__(self, d_model, output_size,
                 dis_feats=[], num_embeds=[], con_feats=[], second_col=None,
                 model_class='gpt2', pre_embed=None, pre_embed_update=False,
                 two_stage=False, lora=True, lora_alpha=32, lora_dim=8, kernel_size=3,
                 wpe_ft=False, semantic_projecting=True, add_poi=True, add_conv_embedder=True,
                 num_meaningful_anchors=30, num_virtual_anchors=15, save_attn_map=False,
                 # ── lora_psp: position-sensitive perturbation dropout on traj embeddings ──
                 # Applied after the conv embedder and before the semantic projector.
                 # 0.0 (default) = nn.Identity() — numerically identical to baseline.
                 # Values in [0.1, 0.25, 0.5] add a dropout mask at that probability.
                 # See hp_study_v3.py for sweep configuration.
                 lora_psp=0.0):

        # ── Model name: only append '-psp{val}' when non-zero ─────────────────
        # This preserves the existing name format for all configs where
        # lora_psp is not set, so saved model file paths are unchanged.
        psp_tag = f'-lpsp{lora_psp}' if lora_psp > 0.0 else ''

        super().__init__(f'LET-d{int(d_model)}-o{int(output_size)}-' + ','.join(map(str, dis_feats + con_feats)) +
                         f'-{model_class}-m{int(num_meaningful_anchors)}-a{int(num_virtual_anchors)}' +
                         (f'-s{second_col}' if second_col is not None else '') +
                         (f'-psp' if semantic_projecting else '') + (f'-twostage' if two_stage else '') +
                         (f'-lora{int(lora_dim)},{int(lora_alpha)}' if lora else '') + (f'-wpeft' if wpe_ft else '') +
                         (f'-conv' if add_conv_embedder else '') + (f'-k{kernel_size}' if kernel_size != 3 else '') +
                         (f'-poi' if add_poi else '') +
                         psp_tag)  # appended last; empty string when lora_psp=0.0

        self.output_size = output_size
        self.dis_feats = dis_feats
        self.con_feats = con_feats
        self.second_col = second_col
        self.model_class = model_class
        self.two_stage = two_stage
        self.add_poi = add_poi

        # ── lora_psp dropout layer ─────────────────────────────────────────────
        # nn.Identity() when lora_psp=0.0 — zero overhead, zero behaviour change.
        # nn.Dropout(p=lora_psp) otherwise — applied in traj_emb().
        if lora_psp > 0.0:
            self.lora_psp_drop = nn.Dropout(p=lora_psp)
        else:
            self.lora_psp_drop = nn.Identity()
        # ──────────────────────────────────────────────────────────────────────

        model_path = get_model_path(model_class)
        self.model_path = model_path
        self.tokenizer = get_tokenizer(model_path, model_class)
        self._init_anchor_words(num_meaningful_anchors, num_virtual_anchors)
        self.encoder, self.emb_size, self.hidden_size, self.word_embedding = \
            get_encoder(model_path, model_class, self.tokenizer,
                        lora=lora, lora_alpha=lora_alpha, lora_dim=lora_dim)

        for i, (name, param) in enumerate(self.encoder.named_parameters()):
            if lora and wpe_ft:
                if 'ln_f' in name or 'wpe' in name: param.requires_grad = True
            elif not lora and wpe_ft:
                if 'ln_f' in name or 'wpe' in name: param.requires_grad = True
                else: param.requires_grad = False
            
            if param.requires_grad:
                print(name, param.requires_grad)

        if semantic_projecting:
            self._init_anchor_embeddings()
            self.pattern_semantic_projector = PatternSemanticProjector(self.emb_size, self.emb_size,
                                                                       meaningful_anchors=self.meaningful_anchors_embeddings,
                                                                       virtual_anchors=self.virtual_anchors_embeddings, n_heads=8,
                                                                       save_attn_map=save_attn_map)
        else:
            self.pattern_semantic_projector = nn.Identity()

        if add_conv_embedder:
            self.embedder = TrajConvEmbedding(self.emb_size, dis_feats, num_embeds, con_feats, kernel_size, pre_embed, pre_embed_update, second_col)
        else:
            self.embedder = TrajEmbedding(self.emb_size, dis_feats, num_embeds, con_feats, pre_embed, pre_embed_update, second_col)

        if self.two_stage:
            self.poi_projector = nn.Linear(self.emb_size, self.emb_size)

        self.out_linear = nn.Sequential(nn.Linear(self.hidden_size, output_size, bias=False),
                                        nn.LayerNorm(output_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(output_size, output_size))

        self.cls_token = nn.Parameter(torch.zeros(self.emb_size).float(), requires_grad=True)
        self._init_prompt_template(lang='zh')

    def _init_anchor_words(self, num_meaningful_anchors, num_virtual_anchors, lang='zh'):
        if lang == 'zh':
            meaningful_words = ["弯", "直", "加", "减", "回", "恒", "停", "快", "慢", "刹", "避", "巡", "绕", "滑", "曲", "稳", "顺", "乱", "粗", "敏", "缓", "静", "动", "超", "滑", "速", "急", "闲", "慎", "莽"]
        elif lang == 'en':
            meaningful_words = ["turn", "straight", "u-turn", "accelerate", "decelerate", "constant", "stop", "fast", "slow", "brake", "swerve", "cruise", "detour", "slide", "zigzag", "steady", "smooth", "erratic", "rough", "agile", "sluggish", "stationary", "dynamic", "overtake", "glide", "rapid", "rushed", "leisurely", "cautious", "reckless"]

        meaningful_words = meaningful_words[:num_meaningful_anchors]
        virtual_words = ["[anchor{}]".format(i) for i in range(num_virtual_anchors)]

        existing_words = set(self.tokenizer.get_vocab().keys())
        added_tokens = list(set(meaningful_words + virtual_words) - existing_words)
        self.tokenizer.add_tokens(added_tokens)

        self.meaningful_anchors = meaningful_words
        self.virtual_anchors = virtual_words

    def _init_anchor_embeddings(self):
        meaningful_anchors = self.tokenizer(self.meaningful_anchors, return_tensors='pt', padding=True) if len(self.meaningful_anchors) > 0 else None
        virtual_anchors = self.tokenizer(self.virtual_anchors, return_tensors='pt', padding=True) if len(self.virtual_anchors) > 0 else None
        
        with torch.no_grad():
            meaningful_anchors_embeddings = self.word_embedding(meaningful_anchors['input_ids']) if meaningful_anchors is not None else None
            virtual_anchors_embeddings = self.word_embedding(virtual_anchors['input_ids']) if virtual_anchors is not None else None

        if isinstance(self.tokenizer, BertTokenizer) or isinstance(self.tokenizer, BertTokenizerFast):
            meaningful_anchors_embeddings = meaningful_anchors_embeddings[:, 1, :] if meaningful_anchors is not None else None
            virtual_anchors_embeddings = virtual_anchors_embeddings[:, 1, :] if virtual_anchors is not None else None
        else:
            meaningful_anchors_embeddings = _mean_pooling(meaningful_anchors_embeddings, meaningful_anchors['attention_mask']) if meaningful_anchors is not None else None
            virtual_anchors_embeddings = _mean_pooling(virtual_anchors_embeddings, virtual_anchors['attention_mask']) if virtual_anchors is not None else None

        self.meaningful_anchors_embeddings = nn.Parameter(meaningful_anchors_embeddings, requires_grad=False) if meaningful_anchors is not None else None
        self.virtual_anchors_embeddings = nn.Parameter(virtual_anchors_embeddings, requires_grad=True) if virtual_anchors is not None else None

    def _init_prompt_template(self, lang='zh'):
        if lang == 'en':
            start_point = "starts near"
            end_point = ", ends near"
            traj = ", passes through"
            suffix_prompt = "This trajectory can be summarized into a word as:"
        else:
            start_point = "起点位于如下地点附近："
            end_point = "终点位于如下地点附近："
            traj = "途径："
            suffix_prompt = "这条轨迹总结成一个字为"
        
        self.start_point, self.end_point, self.traj_template, self.suffix_prompt = \
            self.text_emb(start_point, end_point, traj, suffix_prompt)
        self.start_point, self.end_point, self.traj_template, self.suffix_prompt = \
            [nn.Parameter(e, requires_grad=False) for e in (self.start_point, self.end_point, self.traj_template, self.suffix_prompt)]

    # --- MAIN FORWARD METHOD (FIXED) ---
    def forward(self, *args, **kwargs):
        # Если передан suffix_prompt или token (значит это downstream задача)
        if 'token' in kwargs and kwargs['token'] is not None:
             return self.forward_suffix(*args, **kwargs)
        # Иначе это пре-тренинг или search (без промпта)
        return self.forward_latent(*args, **kwargs)
    # -----------------------------------

    def pretrain_prompt_template(self, o_pois, d_pois, trip, valid_len, start_weekday=1, start_hour=8, recover_type='trip', lang='zh', **kwargs):
        shift_labels = kwargs.get('shift_labels', False)
        head = self._prefix_template(start_weekday, start_hour, lang=lang)
        suffix_prompt = torch.cat([self.suffix_prompt.squeeze(), self.cls_token.unsqueeze(0)], dim=0)

        traj_embeddings = self.traj_emb(trip, valid_len)
        B, L, E_in = traj_embeddings.shape
        start_point, end_point, traj, suffix_prompt = [repeat(x.squeeze(), 'L E -> B L E', B=B) for x in (self.start_point, self.end_point, self.traj_template, suffix_prompt)]

        traj_part, traj_valid_len = concatenate_sequences(traj, traj_embeddings, traj.size(1), valid_len)

        o_placeholder, d_placeholder, trip_placeholder, o_labels, d_labels = None, None, None, None, None
        if self.add_poi:
            o_poi_embeddings, o_valid_len, o_labels = self.poi_emb(o_pois, shift_labels)
            d_poi_embeddings, d_valid_len, d_labels = self.poi_emb(d_pois, shift_labels)

            origin_part, origin_valid_len = concatenate_sequences(start_point, o_poi_embeddings, start_point.size(1), o_valid_len)
            dest_part, dest_valid_len = concatenate_sequences(end_point, d_poi_embeddings, end_point.size(1), d_valid_len)
            poi_part, poi_valid_len = concatenate_sequences(origin_part, dest_part, origin_valid_len, dest_valid_len)

        if self.add_poi:
            if recover_type == 'trip':
                sample, prefix_valid_len = concatenate_sequences(head, poi_part, head.size(1), poi_valid_len)
                sample, sample_valid_len = concatenate_sequences(sample, traj_part, prefix_valid_len, traj_valid_len)
                sample, sample_valid_len = concatenate_sequences(sample, suffix_prompt, sample_valid_len)
                
                # --- FIX: Ensure l is defined ---
                l = sample_valid_len - 1 if shift_labels else sample_valid_len
                # --------------------------------
                
                trip_placeholder = get_batch_mask(B, sample.size(1), l-suffix_prompt.size(1)).long() - get_batch_mask(B, sample.size(1), prefix_valid_len + self.traj_template.size(1)).long() == 1

            elif recover_type == 'poi':
                sample, prefix_valid_len = concatenate_sequences(head, traj_part, head.size(1), traj_valid_len)
                sample, sample_valid_len = concatenate_sequences(sample, poi_part, prefix_valid_len, poi_valid_len)
                sample, sample_valid_len = concatenate_sequences(sample, suffix_prompt, sample_valid_len)
                o_suffix_valid_len = origin_valid_len + head.size(1)
                if shift_labels:
                    o_suffix_valid_len -= 1
                    poi_valid_len -= 1
                o_placeholder = get_batch_mask(B, sample.size(1), o_suffix_valid_len)
                o_placeholder[:, :head.size(1) + self.start_point.size(1)] = False
                d_placeholder = get_batch_mask(B, sample.size(1), poi_valid_len + head.size(1)).long() - get_batch_mask(B, sample.size(1), origin_valid_len + head.size(1) + self.end_point.size(1)).long() == 1
        else:
            sample, sample_valid_len = concatenate_sequences(head, traj_part, head.size(1), traj_valid_len)
            
            # --- FIX: Ensure l is defined ---
            l = sample_valid_len - 1 if shift_labels else sample_valid_len
            # --------------------------------
            
            seperated_placeholder = torch.full((B,), head.size(1) + self.traj_template.size(1), dtype=torch.long).to(sample.device)
            trip_placeholder = get_batch_mask(B, sample.size(1), l).long() - get_batch_mask(B, sample.size(1), seperated_placeholder).long() == 1

        return sample, sample_valid_len, o_placeholder, d_placeholder, trip_placeholder, o_labels, d_labels
    
    def forward_latent(self, x, valid_len, o_pois, d_pois, start_weekday=1, start_hour=8, lang='zh', **kwargs):
        # --- SAFE TRUNCATION ---
        SAFE_TRAJ_LEN = 700
        if x.size(1) > SAFE_TRAJ_LEN:
            x = x[:, :SAFE_TRAJ_LEN, :]
            valid_len = torch.clamp(valid_len, max=SAFE_TRAJ_LEN)
        # -----------------------

        x, valid_len, o_placeholder, d_placeholder, trip_placeholder, o_labels, d_labels = \
            self.pretrain_prompt_template(o_pois, d_pois, x, valid_len, start_weekday, start_hour, lang=lang, **kwargs)
        
        # --- FINAL CLIP ---
        MAX_SEQ_LEN = 1024
        if x.size(1) > MAX_SEQ_LEN:
            x = x[:, :MAX_SEQ_LEN, :]
            valid_len = torch.clamp(valid_len, max=MAX_SEQ_LEN)
            if o_placeholder is not None: o_placeholder = o_placeholder[:, :MAX_SEQ_LEN]
            if d_placeholder is not None: d_placeholder = d_placeholder[:, :MAX_SEQ_LEN]
            if trip_placeholder is not None: trip_placeholder = trip_placeholder[:, :MAX_SEQ_LEN]

        B, L, E_in = x.shape
        batch_mask = get_batch_mask(B, L, valid_len)
        h = x

        output = self.encoder(inputs_embeds=h, attention_mask=batch_mask, output_hidden_states=True)
        h = output.hidden_states[-1]
        h = torch.nan_to_num(h)

        if kwargs.get('recover_type', 'trip') == 'trip':
            return h, trip_placeholder
        else:
            return output.logits, o_placeholder, d_placeholder, o_labels, d_labels
        
    def forward_flip(self, *args, **kwargs):
        h_trip, o_logits, d_logits, o_labels, d_labels = None, None, None, None, None
        
        h_trip, trip_placeholder = self.forward_latent(*args, recover_type='trip', **kwargs)
        h_trip = h_trip[trip_placeholder]

        if self.add_poi:
            logits, o_placeholder, d_placeholder, o_labels, d_labels = self.forward_latent(*args, recover_type='poi', **kwargs)
            o_logits = logits[o_placeholder]
            d_logits = logits[d_placeholder]

        return h_trip, o_logits, d_logits, o_labels, d_labels

    def forward_suffix(self, x, valid_len, o_pois, d_pois, start_weekday, start_hour, suffix_prompt=None, token=None, lang='zh', **kwargs):
        # --- SAFE TRUNCATION FOR DOWNSTREAM ---
        SAFE_TRAJ_LEN = 700
        if x.size(1) > SAFE_TRAJ_LEN:
            x = x[:, :SAFE_TRAJ_LEN, :]
            valid_len = torch.clamp(valid_len, max=SAFE_TRAJ_LEN)
        # --------------------------------------

        # --- IMPORTANT: Use self.pretrain_prompt_template (or just copy logic if needed) ---
        # The original code used a shared logic. Here we adapt prompt_template to use token if provided.
        # But wait, original code had a separate prompt_template method? No, it seems it reused logic.
        # Let's define a prompt_template that handles the downstream token insertion correctly.
        
        res_sample, res_valid_len = self.prompt_template_logic(o_pois, d_pois, x, valid_len, start_weekday, start_hour, suffix_prompt, token, lang=lang, **kwargs)

        MAX_SEQ_LEN = 1024
        if res_sample.size(1) > MAX_SEQ_LEN:
            res_sample = res_sample[:, :MAX_SEQ_LEN, :]
            res_valid_len = torch.clamp(res_valid_len, max=MAX_SEQ_LEN)

        B, L, E_in = res_sample.shape
        batch_mask = get_batch_mask(B, L, res_valid_len)
        h = res_sample

        output = self.encoder(inputs_embeds=h, attention_mask=batch_mask, output_hidden_states=True)
        h = output.hidden_states[-1]
        h = torch.nan_to_num(h)
        output = self.out_linear(h[:, -1])

        return output

    def prompt_template_logic(self, o_pois, d_pois, trip, valid_len, start_weekday=1, start_hour=8, suffix_prompt=None, token=None, lang='zh', **kwargs):
        """ Helper to construct the input sequence for downstream tasks """
        B, L, _ = trip.shape
        head = self._prefix_template(start_weekday, start_hour, lang=lang)
        
        # Embed suffix prompt text
        suffix_emb = self.text_emb(suffix_prompt)[0].squeeze() if suffix_prompt else self.suffix_prompt.squeeze()
        
        # Append the learnable token
        if token is not None:
             suffix_prompt_full = torch.cat([suffix_emb, token.unsqueeze(0)], dim=0)
        else:
             suffix_prompt_full = torch.cat([suffix_emb, self.cls_token.unsqueeze(0)], dim=0)

        traj_embeddings = self.traj_emb(trip, valid_len)
        
        # Repeat for batch
        traj, suffix_prompt_batch = [repeat(x.squeeze(), 'L E -> B L E', B=B) for x in (self.traj_template, suffix_prompt_full)]
        
        traj_part, traj_valid_len = concatenate_sequences(traj, traj_embeddings, traj.size(1), valid_len)

        if self.add_poi:
            o_poi_embeddings, o_valid_len, _ = self.poi_emb(o_pois)
            d_poi_embeddings, d_valid_len, _ = self.poi_emb(d_pois)
            start_point, end_point = [repeat(x.squeeze(), 'L E -> B L E', B=o_poi_embeddings.size(0)) for x in (self.start_point, self.end_point)]
            
            origin_part, origin_valid_len = concatenate_sequences(start_point, o_poi_embeddings, start_point.size(1), o_valid_len)
            dest_part, dest_valid_len = concatenate_sequences(end_point, d_poi_embeddings, end_point.size(1), d_valid_len)
            
            sample, prefix_valid_len = concatenate_sequences(head, origin_part, head.size(1), origin_valid_len)
            if not kwargs.get('d_mask', False):
                sample, prefix_valid_len = concatenate_sequences(sample, dest_part, prefix_valid_len, dest_valid_len)
        else:
            sample, prefix_valid_len = head, head.size(1)

        sample, prefix_valid_len = concatenate_sequences(sample, traj_part, prefix_valid_len, traj_valid_len)
        res_sample, res_valid_len = concatenate_sequences(sample, suffix_prompt_batch, prefix_valid_len)

        return res_sample, res_valid_len

    def text_emb(self, *texts, return_valid_len=False, shift_labels=False):
        texts = [self.tokenizer(x, return_tensors='pt', padding=True, truncation=True).to(self.cls_token.device) for x in texts]
        batch_mask = [x['attention_mask'] for x in texts]
        valid_lens = [e.sum(dim=1)-2 for e in batch_mask]
        if return_valid_len:
            labels = []
            for poi, l in zip(texts[0]['input_ids'], valid_lens[0]):
                labels.append(poi[2:l+1]) if shift_labels else labels.append(poi[1:l+1])
            labels = torch.cat(labels, dim=0).long()
        with torch.no_grad():
            if isinstance(self.tokenizer, BertTokenizer) or isinstance(self.tokenizer, BertTokenizerFast):
                texts = [self.word_embedding(x['input_ids'][:, 1:-1]) for x in texts]
            else:
                texts = [self.word_embedding(x) for x in texts]
        if return_valid_len:
            return texts, valid_lens, labels
        return texts

    def poi_emb(self, pois, shift_labels=False):
        valid_len, labels = None, None
        if not self.two_stage:
            pois, valid_len, labels = self.text_emb(pois, return_valid_len=True, shift_labels=shift_labels)
            pois, valid_len = pois[0], valid_len[0]
        else:
            pois = self.poi_projector(pois)
        return pois, valid_len, labels

    def traj_emb(self, x, valid_len):
        x = torch.where(get_batch_mask(x.shape[0], x.shape[1], valid_len).unsqueeze(-1), x, torch.zeros_like(x))
        x = self.embedder(x)
        # ── lora_psp dropout ──────────────────────────────────────────────────
        # Applied after the convolutional embedder and before the semantic
        # projector. This is the position-sensitive location: the conv embedder
        # has already encoded local trajectory structure, so dropout here acts
        # on spatially-structured features rather than raw inputs.
        # When lora_psp=0.0, self.lora_psp_drop is nn.Identity() — no change.
        x = self.lora_psp_drop(x)
        # ─────────────────────────────────────────────────────────────────────
        x = self.pattern_semantic_projector(x)
        return x

    def _prefix_template(self, start_weekday, start_hour, lang='zh'):
        start_weekday, start_hour = start_weekday.long(), start_hour.long()
        if lang == 'en':
            weekday = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            prefix = [f'This trajectory happened at {h} o\'clock on {weekday[i]}, ' for i, h in zip(start_hour, start_weekday)]
        else:
            weekday = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
            prefix = [f'这条轨迹发生于{weekday[i]}的{h}点，' for i, h in zip(start_weekday, start_hour)]
        prefix = torch.concat(self.text_emb(*prefix))
        return prefix

    def save_model(self):
        self.encoder.save_pretrained(self.model_path)

    def load_model(self):
        self.encoder.load_adapter(self.model_path)
