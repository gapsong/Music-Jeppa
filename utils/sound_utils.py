import math
import numpy as np
from numpy.lib.function_base import append
import torch as t
import torch.nn.functional as F
import torchaudio.transforms
from miniaudio import (
    mp3_stream_file, 
    stream_file,
    convert_sample_format,
    SampleFormat,
    ffi, lib,
    _get_filename_bytes, _create_int_array, DecodeError
)
import lameenc

def read_audio_file(filename):
    pass

def read_mp3_file(filename, seek_frame: int = 0, frames_to_read: int = -1):
    samples = None
    filenamebytes = _get_filename_bytes(filename)
    sample_rate, nchannels = None, None

    def _read_from_buffer(_mp3, _frames_to_read, _buf_ptr, _decodebuffer):
        num_samples = lib.drmp3_read_pcm_frames_s16(_mp3, _frames_to_read, _buf_ptr)
        if num_samples <= 0:
            return None
        buffer = ffi.buffer(_decodebuffer, num_samples * 2 * _mp3.channels)
        samples = _create_int_array(2)
        samples.frombytes(buffer)
        return samples

    whole_file = frames_to_read is None or frames_to_read < 1
    if whole_file:
        frames_to_read = 1024

    with ffi.new("drmp3 *") as mp3:
        if not lib.drmp3_init_file(mp3, filenamebytes, ffi.NULL):
            raise DecodeError("could not open/decode file")

        sample_rate, nchannels = mp3.sampleRate, mp3.channels
        if seek_frame > 0:
            result = lib.drmp3_seek_to_pcm_frame(mp3, seek_frame)
            if result <= 0:
                raise DecodeError("can't seek")
        try:
            with ffi.new("drmp3_int16[]", frames_to_read * mp3.channels) as decodebuffer:
                buf_ptr = ffi.cast("drmp3_int16 *", decodebuffer)
                if not whole_file:
                    samples = _read_from_buffer(mp3, frames_to_read, buf_ptr, decodebuffer)
                else:
                    while True:
                        tmp = _read_from_buffer(mp3, frames_to_read, buf_ptr, decodebuffer)
                        if tmp is not None:
                            if samples is None:
                                samples = tmp
                            else:
                                samples.extend(tmp)
                        else:
                            break
        finally:
            lib.drmp3_uninit(mp3)

    samples = convert_sample_format(
        SampleFormat.SIGNED16, bytes(samples), SampleFormat.FLOAT32
    )
    samples = np.transpose(
        np.frombuffer(samples, dtype=np.float32).reshape((-1, nchannels))
    )

    return samples, sample_rate

def save_mp3_file(audio, out_path, sample_rate=44100, bitrate=128, mono=True, quality=2):
    # audio: [cs, ts] in float32
    audio = np.transpose(audio).reshape(audio.shape[0] * audio.shape[1])
    audio = convert_sample_format(
        SampleFormat.FLOAT32, audio.tobytes(), SampleFormat.SIGNED16
    )
    audio = np.frombuffer(audio, dtype=np.int16)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(1 if mono else 2)
    encoder.set_quality(quality)  # 2-highest, 7-fastest
    mp3data = encoder.encode(audio)
    mp3data += encoder.flush()
    with open(out_path, "wb") as f:
        f.write(mp3data)



# feats.size() = [B, H, T]
# See: https://arxiv.org/pdf/1606.00021.pdf
def compute_gram_matrix(feats, layer_id, method='spec', params=None, ema_alpha=0.9, ema_block=200):
    if method == 'spec':
        spec = compute_spectogram(feats, params, layer_id)
        a, b, c, d = spec.size()
        x = spec.unsqueeze(1).expand(a, b, b, c, d)
        y = spec.unsqueeze(2).expand(a, b, b, c, d)
        return (x * y).mean(dim=-1)
    elif method == 'spec_ema':
        spec = compute_spectogram(feats, params)
        a, b, c, d = spec.size()
        spec = compute_ema(spec.view(a * b * c, d), alpha=ema_alpha, block_size=ema_block).view(a, b, c, d)
        x = spec.unsqueeze(1).expand(a, b, b, c, d)
        y = spec.unsqueeze(2).expand(a, b, b, c, d)
        return (x * y).mean(dim=-1)
    elif method == 'spec_freq_ema':
        spec = compute_spectogram(feats, params)
        a, b, c, d = spec.size()
        spec = spec.permute(0,1,3,2) # B, H, T, F
        spec = compute_ema(spec.reshape(a * b * d, c), alpha=ema_alpha, block_size=ema_block).view(a, b, d, c)
        spec = spec.permute(0,1,3,2) # B, H, F, T
        x = spec.unsqueeze(1).expand(a, b, b, c, d)
        y = spec.unsqueeze(2).expand(a, b, b, c, d)
        return (x * y).mean(dim=-1)
    elif method == 'spec_feat_ema':
        a, b, c = feats.size()
        feat_ema = compute_ema(feats.view(a * b, c), alpha=ema_alpha, block_size=ema_block).view(a, b, c)
        return compute_gram_matrix(feat_ema, method='spec', params=params)
    elif method == 'fft':
        ffts = t.fft.fft(feats, dim=2, norm='backward')
        ffts_abs = t.abs(ffts)
        return t.bmm(ffts_abs, ffts_abs.permute(0,2,1)) / ffts_abs.size(1)
    elif method == 'fft_sep':
        ffts = t.fft.fft(feats, dim=2, norm='backward')
        ffts_abs = t.abs(ffts) # [B, Z, F]
        # mini = t.min(ffts_abs)
        # maxi = t.max(ffts_abs)
        # meani = t.mean(ffts_abs)
        # ffts_abs = t.clip(ffts_abs, 0, 100)
        # ffts_abs = ffts # [B, Z, F]
        # ffts_abs = t.sqrt(ffts_abs)
        # ffts_abs = ffts_abs.permute(0,2,1)
        # ffts_abs = F.softmax(ffts_abs + 10000000.0,dim=2) * 1000000.0
        # ffts_abs = ffts_abs.permute(0,2,1)
        # ffts_abs = softmax_nd(ffts_abs.double() + 100000.0) * 10000000.0
        # minni = t.min(ffts_abs)
        ffts_abs = t.log1p(ffts_abs)
        # ffts_abs = F.log()
        a, b, c = ffts_abs.size()
        x = ffts_abs.unsqueeze(1).expand(a, b, b, c)
        y = ffts_abs.unsqueeze(2).expand(a, b, b, c)
        return (x * y)
        return t.log1p(x * y * 0.02)
    elif method == 'direct':
        return t.bmm(feats, feats.permute(0,2,1)) / feats.size(2)


# [B, D1, D2, ...]
def softmax_nd(x, use_log=True):
    s = x.size()
    x_flat = t.flatten(x, start_dim=1)
    if use_log:
        x_flat = F.log_softmax(x_flat, dim=1)
        # x_flat = t.log(1.5 + F.softmax(x_flat, dim=1))
    else:
        x_flat = F.softmax(x_flat, dim=1)
    return x_flat.view(s)


# x.shape = [B, F, T]
# Computes log(||STFT(x)|| + 1)
def compute_spectogram(x, params, layer_id, smooth_feats=False, smooth_param=1.0):
    assert len(x.shape) == 3
    y = x.view(x.shape[0] * x.shape[1], x.shape[2])
    y = t.stft(y, 
        params.n_fft, 
        params.hop_length, 
        win_length=params.window_size, 
        window=t.hann_window(params.window_size, device=y.device),
        normalized=False,
        center=True
    )
    y = t.norm(y, p=2, dim=-1)
    y = torchaudio.transforms.MelScale(n_mels=int(256), sample_rate=int(44100 / (2**layer_id)), n_stft=y.shape[1]).cuda()(y)
    y = y.view(x.shape[0], x.shape[1], y.shape[1], y.shape[2])
    # y = t.log1p(y)
    return y


# Note: This function does NOT normalize the data
# data: [B, T] or [B, C, T]
def compute_ema(data, alpha=0.95, initial_value=None, block_size=100, with_variance=False, gpu=True):
    is_numpy = not t.is_tensor(data)
    if is_numpy:
        data = t.from_numpy(data)

    data_shape = len(data.size())
    if data_shape == 2:
        bs, ts = data.size()
    elif data_shape == 3:
        bs, hs, ts = data.size()
        data = data.reshape(bs * hs, ts)

    device = data.device
    if gpu:
        data = data.cuda()

    if initial_value is None:
        initial_value = data[:, 0, None]

    size = ts if ts < block_size else block_size
    a = t.full((size,), alpha, device=data.device, dtype=t.float64)
    exps = t.arange(0, size, device=data.device, dtype=t.float64).flip(dims=(0,))
    log_a = t.log(a) * exps
    a = t.exp(log_a)
    # a = t.pow(a, exps)

    batches = []
    for i in range(math.ceil(ts / size)):
        offset = size if data.shape[1] >= (i+1) * size else data.shape[1] - (i*size)
        emas = data.type(dtype=t.float64)[:, i*size:(i*size)+offset] * a[None, -offset:] * (1-alpha)
        emas = t.cumsum(emas, dim=1)
        emas = emas + t.pow(t.tensor(alpha), offset) * initial_value
        emas = emas / a[None, -offset:]
        batches.append(emas)
        initial_value = emas[:, -1, None]

    ema = t.cat(batches, dim=1).type(t.float32)

    if with_variance:

        #   a * (emvars[-1] + (1-a) * (sample - emas[-2]) * (sample - emas[-2]))
        # = a * emvars[-1] + a * (1-a) * (sample - emas[-2]) * (sample - emas[-2])
        # = a * emvars[-1] + (1-a) * [a * (sample**2 - 2*sample*emas[-2] + emas[-2]**2)]
        # = compute_ema(data = [a * (sample**2 - 2*sample*emas[-2] + emas[-2]**2)])
        padded_ema = t.cat((
            t.zeros(ema.shape[0], 1, device=ema.device, dtype=ema.dtype),
            ema[:, :-1]
        ), dim=1)
        # testo = padded_ema[:, 0]
        var_data = alpha * (data**2 - 2 * data * padded_ema + padded_ema**2)
        emvar = compute_ema(var_data, alpha, block_size=block_size, with_variance=False)

    if data_shape == 3:
        ema = ema.reshape(bs, hs, ts)

    if is_numpy:
        ema = ema.cpu().numpy()
    else:
        ema = ema.to(device)


    if with_variance:
        if data_shape == 3:
            emvar = emvar.reshape(bs, hs, ts)

        if is_numpy:
            emvar = emvar.cpu().numpy()
        else:
            emvar = emvar.to(device)
        return ema, emvar
    return ema



def compute_ema_old(data, alpha=0.95, initial_value=None):
    if initial_value is None:
        initial_value = data[:, 0]
    emas = []
    # emas.append(t.full_like(initial_value, initial_value, device='cuda'))  # TODO: when not using normalized d anymore, use d_norm
    emas.append(initial_value)  # TODO: when not using normalized d anymore, use d_norm

    for i in range(1, data.shape[1]):
        sample = data[:, i]
        emas.append((alpha * emas[-1]) + ((1 - alpha) * sample))
    emas = t.stack(emas, dim=1)
    return emas


def compute_gram_from_spec(spec):
    a = spec.unsqueeze(1).expand(spec.shape[0], spec.shape[1], spec.shape[1], spec.shape[2], spec.shape[3])
    b = spec.unsqueeze(2).expand(spec.shape[0], spec.shape[1], spec.shape[1], spec.shape[2], spec.shape[3])
    gram = (a * b).sum(dim=-1)
    return gram

# feats [B, F, T]
def compute_gram_from_feats(feats, smooth_feats=True):
    if smooth_feats:
        bs, hs, ts = feats.shape[0], feats.shape[1], feats.shape[2]
        feats = feats.reshape(bs, hs * ts)
        feats = F.softmax(feats, dim=1)
        feats = feats.view(bs, hs, ts)
    lolol = t.bmm(feats, feats.permute(0,2,1))
    return lolol
    a = feats.unsqueeze(1).expand(feats.shape[0], feats.shape[1], feats.shape[1], feats.shape[2])
    b = feats.unsqueeze(2).expand(feats.shape[0], feats.shape[1], feats.shape[1], feats.shape[2])
    gram = (a * b).sum(dim=-1)
    return gram