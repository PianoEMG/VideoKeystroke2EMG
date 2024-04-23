import torch
import torch.nn as nn
import math

def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

# Alignment Bias
def enc_dec_mask(device, dataset, T, S):
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    elif dataset == "Beat":
        for i in range(T):
            mask[i, i] = 0
    return (mask==1).to(device=device)

# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=120, max_seq_len=1024):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Audio2EMG(nn.Module):
    def __init__(self, args):
        super(Audio2EMG, self).__init__()
        self.device = args.device
        self.audio_feature_map = nn.Linear(768, args.feature_dim)
        self.emg_feature_map = nn.Linear(9, args.feature_dim)
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period = args.period)
        self.biased_mask = init_biased_mask(n_head = 4, max_seq_len = 1024, period=args.period).repeat(2, 1, 1)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=4, dim_feedforward=2*args.feature_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        self.emg_decoder = nn.Linear(args.feature_dim, 9)

        nn.init.constant_(self.emg_decoder.weight, 0)
        nn.init.constant_(self.emg_decoder.bias, 0)

    def forward(self, audio, emg, criterion):
        audio_feat = self.audio_feature_map(audio)
        # print(emg[:,:-1].shape)
        # print(emg[:,0].unsqueeze(1).shape)
        # print(torch.zeros_like(emg[:,0].unsqueeze(1)).shape)
        emg_input = torch.cat((torch.zeros_like(emg[:,0].unsqueeze(1)), emg[:,:-1]), dim=1)
        # print(emg_input.shape)
        emg_input = self.emg_feature_map(emg_input)
        emg_input = self.PPE(emg_input)
        tgt_mask = self.biased_mask[:, :emg_input.shape[1], :emg_input.shape[1]].clone().detach().to(device=self.device)
        memory_mask = enc_dec_mask(self.device, 'Beat', emg_input.shape[1], audio_feat.shape[1])
        # print(emg_input.shape, audio_feat.shape, tgt_mask.shape, memory_mask.shape)
        feat_out = self.transformer_decoder(emg_input, audio_feat, tgt_mask=tgt_mask, memory_mask=memory_mask)
        feat_out = self.emg_decoder(feat_out)

        loss = criterion(feat_out, emg)

        return loss, feat_out
    
    def predict(self, audio):
        frame_num = audio.shape[1]
        audio_feat = self.audio_feature_map(audio)

        for i in range(frame_num):
            if i == 0:
                emg_input = torch.zeros_like(audio_feat[:, 0, :9]).unsqueeze(1)
                emg_input = self.PPE(emg_input)
            else:
                emg_input = self.PPE(emg_input)

            tgt_mask = self.biased_mask[:, :emg_input.shape[1], :emg_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, emg_input.shape[1], audio_feat.shape[1])
            feat_out = self.transformer_decoder(emg_input, audio_feat, tgt_mask=tgt_mask, memory_mask=memory_mask)
            feat_out = self.emg_decoder(feat_out)
