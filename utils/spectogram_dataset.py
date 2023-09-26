from dataclasses import dataclass
import math
from pathlib import Path
import librosa
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import torch

@dataclass
class SpecFileInfo():
    path: Path
    duration: float

class SpectogramDataset(Dataset):
    def __init__(
        self, 
        files, 
        sample_len=None, 
        min_duration=None,
        max_duration=None,
        aug_shift=True,
        spec_dims=128,
        spec_dtype=np.float16,
        frames_per_sec=100, 
        unit="seconds",
        least_sample_len_divisor=1,
        return_info=False,
        seqlen_dim=0,
        filter_fn = None,
        normalize = False,
        specs_mean = None,
        specs_std = None,
    ) -> None:
        super().__init__()

        # Convert sample length units to frames if necessary
        self.sample_len = (
            None 
            if sample_len is None or (sample_len < 1)
            else (int(sample_len * frames_per_sec) if unit == "seconds" else sample_len)
        )

        assert least_sample_len_divisor is None or isinstance(least_sample_len_divisor, int)
        self.sample_len_ld = least_sample_len_divisor
        if self.sample_len is not None and least_sample_len_divisor is not None:
            self.sample_len = (self.sample_len // self.sample_len_ld) * self.sample_len_ld

        # Set min and max sample_len
        self.min_duration = (
            (self.sample_len if self.sample_len is not None else 1) 
            if min_duration is None else 
            (int(min_duration * frames_per_sec) if unit == "seconds" else min_duration)
        )
        self.max_duration = (
            math.inf 
            if max_duration is None else 
            (int(max_duration * frames_per_sec) if unit == "seconds" else max_duration)
        )

        # Misc
        self.spec_dims = spec_dims
        self.spec_dtype = spec_dtype
        self.bytes_per_entry = 2 if spec_dtype == np.float16 else 4
        self.frames_per_sec = frames_per_sec
        self.supported_data_types = ('npy', )
        self.ret_info = return_info
        self.aug_shift = aug_shift
        self.seqlen_dim = seqlen_dim
        self.filter_fn = filter_fn
        self.normalize = normalize

        self.check_and_process_files(files)


    
    def check_and_process_files(self, files):
        # Load list of file candidates
        if isinstance(files, str):
            files = [files]
        else:
            files = list(files)
        
        keep = []
        while len(files) > 0:
            f = files.pop()
            p = Path(f)
            if p.is_file() and p.suffix[1:] in self.supported_data_types:
                keep.append(p)
            elif p.is_dir():
                files += librosa.util.find_files(str(p), self.supported_data_types)
        files = keep
        print(f"Found {len(files)} spectogram file candidates")

        # Filter files
        self.infos = []
        for f in files:
            if self.filter_fn is not None and not self.filter_fn(f.stem):
                continue

            try:
                if f.suffix == ".npy":
                    dur = self.get_npy_duration(f)
                elif f.suffix == ".spec":
                    dur = self.get_spec_file_duration(f)
            except:
                continue

            if dur < self.min_duration or dur > self.max_duration:
                continue

            # finally add
            self.infos.append(SpecFileInfo(
                path = f,
                duration = dur,
            ))
        
        self.cumsum_durs = np.cumsum(np.array([info.duration for info in self.infos]))
        print(f"Keeping {len(self.infos)} spectograms")
        print(f"Total duration: {self.cumsum_durs[-1] / (3600 * self.frames_per_sec)} hours")
        print(f"Avg duration: {self.cumsum_durs[-1] / (self.frames_per_sec * len(self.cumsum_durs))} secs")

    def load_spec_file(self, file, offset=0, duration=None):
        # offset and duration is in tokens, so need to convert to bytes
        # TODO: needs testing!
        if self.seqlen_dim == 0:
            offset_bts = offset * self.spec_dims * self.bytes_per_entry
            if duration is not None:
                duration_bts = duration * self.spec_dims * self.bytes_per_entry
        else:
            pass
            # TODO: this will be harder...

        with open(file, mode='br') as specfile:
            specfile.seek(offset_bts)
            if duration is None:
                data = specfile.read()
            else:
                data = specfile.read(duration_bts)
            spec = np.frombuffer(data, dtype=self.spec_dtype).reshape(-1, self.spec_dims)
        return spec

    def get_npy_duration(self, file, unit="tokens"):
        dur = np.load(file, mmap_mode='r').shape[self.seqlen_dim]
        if unit == "seconds":
            dur = dur / self.frames_per_sec
        return dur

    def get_spec_file_duration(self, file, unit="tokens"):
        # TODO: check correctness
        if not isinstance(file, Path):
            file = Path(file)
        sz = file.stat().st_size
        dur = int(sz / (self.spec_dims * self.bytes_per_entry))
        if unit == "seconds":
            dur = dur / self.frames_per_sec
        return dur

    def load_spectogram(self, item, offset=0, num_tokens=None):
        # Get info
        info = self.infos[item]
        
        num_tokens = num_tokens or info.duration

        # TODO: allow padding
        assert offset >= 0 and offset <= (info.duration-num_tokens) and num_tokens > 0

        if self.sample_len_ld is not None:
            num_tokens = (num_tokens // self.sample_len_ld) * self.sample_len_ld

        if info.path.suffix == ".spec":
            spec = self.load_spec_file(
                str(info.path),
                offset = offset,
                duration = num_tokens
            )
        elif info.path.suffix == ".npy":
            spec = np.load(str(info.path), mmap_mode='c')[offset:(offset+num_tokens)]
        
        # normalize (to 0.5 std)
        if self.normalize:
            if self.specs_mean is not None and self.specs_std is not None:
                mean, std = self.specs_mean, self.specs_std
            else:
                mean, std = np.mean(spec), np.std(spec)
            spec = (spec - mean) / (2 * std)

        return spec


    # TODO: this method should be outsourced since it exists in every seq dataset
    def get_index_offset(self, item):
        if self.sample_len is None:
            return item, 0
        else:
            half_interval = int(self.sample_len / 2)
            shift = np.random.randint(-half_interval, half_interval) if self.aug_shift else 0

            # get sample offset (optionally + aug shift)
            offset = item * self.sample_len + shift # Note we centred shifts, so adding now

            # compute midpoint
            midpoint = offset + half_interval
            cs = self.cumsum_durs
            assert 0 <= midpoint < cs[-1], f'Midpoint {midpoint} of item beyond total length {cs[-1]}'

            # find song index by searching over cumsum
            index = np.searchsorted(cs, midpoint)  # index <-> midpoint of interval lies in this song
            start, end = cs[index - 1] if index > 0 else 0.0, cs[index] # start and end of current song
            assert start <= midpoint <= end, f"Midpoint {midpoint} not inside interval [{start}, {end}] for index {index}"

            # optionally adjust offset, if sample boundaries lie outside of the song
            if offset > end - self.sample_len: # Going over song
                offset = max(start, offset - half_interval - 1)  # Now should fit
            elif offset < start: # Going under song
                offset = min(end - self.sample_len, offset + half_interval)  # Now should fit
            assert start <= offset <= end - self.sample_len, f"Offset {offset} not in [{start}, {end - self.sample_len}]. End: {end}, SL: {self.sample_len}, Index: {index}"

            offset = int(offset - start)
            return index, offset

    # TODO: same for this
    def __len__(self):
        if self.sample_len is None:
            return len(self.infos)
        else:
            return int(np.floor(self.cumsum_durs[-1] / self.sample_len))

    def __getitem__(self, item):
        if item >= len(self): raise IndexError
        idx, offset = self.get_index_offset(item)
        spec = self.load_spectogram(idx, offset, self.sample_len)
        if not self.ret_info:
            return spec
        else:
            return spec, self.infos[idx]

    def average_spec_len(self):
        return sum([i.duration for i in self.infos]) / len(self.infos)


    @staticmethod
    def collater(batch):
        """ Crops to smallest length, if specs have different lengths """
        # TODO: check shape of specs
        min_len = min([b.shape[0] for b in batch])
        # max_len = max([b.shape[0] for b in batch])
        cropped = torch.from_numpy(np.stack([b[:min_len] for b in batch]))
        # print(f"min: {min_len}, max: {max_len}")
        return cropped


def test():
    # # loading
    # import torch as t
    # import jukebox.utils.io as jbio
    # import torchaudio.compliance.kaldi as kaldi
    # waveform1, sr1 = jbio.load_audio(
    #     '/home/tommi/datasets/mpf/audio/3500.mp3',
    #     sr = 44100
    # )
    # waveform2, sr2 = jbio.load_audio(
    #     '/home/tommi/datasets/mpf/audio/3500.mp3',
    #     sr = 16000
    # )
    # waveform1 = t.from_numpy(waveform1 - waveform1.mean())
    # waveform2 = t.from_numpy(waveform2 - waveform2.mean())
    # spec1: t.tensor = kaldi.fbank(
    #     waveform1,
    #     htk_compat=True,
    #     sample_frequency=sr1,
    #     use_energy=False,
    #     window_type='hanning',
    #     num_mel_bins=128,
    #     dither=0.0,
    #     frame_shift=10
    # ).cpu().numpy().astype(np.float16)

    # spec2: t.tensor = kaldi.fbank(
    #     waveform2,
    #     htk_compat=True,
    #     sample_frequency=sr2,
    #     use_energy=False,
    #     window_type='hanning',
    #     num_mel_bins=128,
    #     dither=0.0,
    #     frame_shift=10
    # ).cpu().numpy().astype(np.float16)

    # difffii = ((spec1-spec2) ** 2).max()

    # spec0 = load_spec_file('/home/tommi/datasets/mpf/fbanks/fbanks/audio/3500.mp3.spec')
    # spec1 = load_spec_file('/home/tommi/datasets/mpf/fbanks/fbanks/audio/3500.mp3.spec', offset=150)
    # spec2 = load_spec_file('/home/tommi/datasets/mpf/fbanks/fbanks/audio/3500.mp3.spec', duration=34)
    # spec3 = load_spec_file('/home/tommi/datasets/mpf/fbanks/fbanks/audio/3500.mp3.spec', offset=150, duration=34)

    # diff = spec0 - spec
    # mini, maxi, meani = diff.min(), diff.max(), diff.mean()
    # diff = spec[150:] - spec1
    # mini, maxi, meani = diff.min(), diff.max(), diff.mean()
    # diff = spec[:34] - spec2
    # mini, maxi, meani = diff.min(), diff.max(), diff.mean()
    # diff = spec[150:(150+34)] - spec3
    # mini, maxi, meani = diff.min(), diff.max(), diff.mean()

    # dataset
    dataset = SpectogramDataset(
        '/home/tommi/datasets/mpf/mpf_spec/',
        sample_len=1024,
        unit="tokens"
    )
    print(f'Len: {len(dataset)}')
    import time
    t0 = time.time()
    for i, d in enumerate(dataset):
        print(f"{i}")
        pass
        # print(f"{i}")
    print(f"Dur: {time.time() - t0}")

def load_spec_file(file, offset=0, duration=None):
    offset_bts = offset * 128 * 2
    if duration is not None:
        duration_bts = duration * 128 * 2

    with open(file, mode='br') as specfile:
        specfile.seek(offset_bts)
        if duration is None:
            data = specfile.read()
        else:
            data = specfile.read(duration_bts)
        spec = np.frombuffer(data, dtype=np.float16).reshape(-1, 128)
    return spec


# test()