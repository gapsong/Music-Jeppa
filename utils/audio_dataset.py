from importlib.resources import path
from typing import Tuple
# import av
import io
import time
from dataclasses import dataclass
import math
from pathlib import Path
from threading import Lock
import zipfile
import librosa
import pydub
from pydub.utils import mediainfo
from pydub import AudioSegment
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio.compliance.kaldi as kaldi
import torchaudio
from torchaudio.transforms import Resample
from utils.sound_utils import read_mp3_file
import json

@dataclass
class AudioFileInfo():
    path: Path
    metadata: dict
    duration: int
    zipfile_idx: str

@dataclass
class LoadedAudioFileInfo():
    path: Path
    metadata: dict
    duration: int
    snippet_offset: float
    snippet_duration: float
    zipfile_idx: str

def torch_metadata_from_dict(d):
    return torchaudio.backend.common.AudioMetaData(
        sample_rate = int(d['sample_rate']),
        num_frames = int(d['num_frames']),
        num_channels = int(d['num_channels']),
        bits_per_sample = int(d['bits_per_sample']),
        encoding = d['encoding'],
    )

class AudioDataset(Dataset):
    def __init__(self, 
                 files,
                 sample_rate=44100, 
                 sample_len=None,
                 min_duration=None, max_duration=None,
                 unit='seconds',
                 mono=True,
                 center=True,
                 aug_shift=True,
                 filter_fn=None,
                 return_filenames=False,
                 return_info=False,
                 create_metadata_files=True,
                #  format="f32", # [f32, s16, mel, cqt]
                #  format_options=None,

                 # TODO: deprecate spec arguments
                 to_spectogram=False,
                 spec_type="vqt",
                 spec_mel_bins=120,
                 spec_frame_length=512,
                 spec_frame_shift=10,
                 spec_bins_per_octave=12,
                 spec_fmin=16.35,
                 spec_use_power=True,
                 least_sample_len_divisor=1,
                 debug = False,
    ):
        super().__init__()

        assert unit == 'seconds' or unit == 'frames'
        assert unit == 'seconds' or sample_rate is not None
        assert unit == 'seconds' or isinstance(sample_len, int)
        assert isinstance(sample_rate, int)

        self.unit = unit # seconds or frames

        self.sample_len = sample_len if sample_len is not None and sample_len > 0 else None
        self.least_sl_div = least_sample_len_divisor

        if self.sample_len is not None and self.least_sl_div is not None:
            assert self.sample_len % self.least_sl_div == 0

        self.min_duration = (
            min_duration
            if min_duration is not None else 
            (self.sample_len if self.sample_len is not None else 1) 
        )
        self.max_duration = (
            max_duration
            if max_duration is not None else 
            math.inf 
        )

        self.sr = sample_rate if sample_rate > 0 else None

        # Misc
        self.supported_audio_types = ['mp3', 'wav', 'flac', 'm4a']
        self.aug_shift = aug_shift
        self.center = center
        self.ret_names = return_filenames
        self.ret_info = return_info
        self.padding = False # TODO
        self.mono = mono
        self.filter_fn = filter_fn
        self.create_metadata_files = create_metadata_files
        self.zipfiles = {}
        self.format = 'f32'
        self.format_options = None

        # For specs
        self.to_spec = to_spectogram
        self.spec_type = spec_type
        self.spec_mel_bins = spec_mel_bins
        self.spec_frame_length = spec_frame_length
        self.spec_frame_shift = spec_frame_shift
        self.spec_bins_per_octave = spec_bins_per_octave
        self.spec_fmin = spec_fmin
        self.spec_use_power = spec_use_power

        self.check_and_process_files(files)
        self._lock = Lock() # for multiprocess zipfile access

        if debug:
            self._debug = {
                'audio_load_total_dur': 0,
                'audio_resample_total_dur': 0,
                'audio_mono_total_dur': 0,
                'audio_center_total_dur': 0,
                'spec_calc_total_dur': 0,
            }
        else:
            self._debug = None

    def check_and_process_files(self, files):
        # Load list of file candidates
        if isinstance(files, str):
            files = [files]
        else:
            files = list(files)
        durs = []
        keep = []
        zipped_files = set()
        zipped_to_archive = {}
        while len(files) > 0:
            f = files.pop()
            p = Path(f)
            if p.is_file() or p in zipped_files:
                if p.suffix[1:] in self.supported_audio_types:
                    keep.append(p)  
                elif p.suffix == ".zip":
                    # zips.append(p)
                    zf = zipfile.ZipFile(str(p), mode='r')
                    infolist = zf.infolist()
                    zfiles = [Path(zi.filename) for zi in infolist]
                    zipped_files.update(zfiles)
                    zipped_to_archive.update({k: str(p) for k in zfiles})
                    files += zfiles
                    self.zipfiles[str(p)] = zf
            elif p.is_dir():
                files += librosa.util.find_files(str(p))
        files = keep

        print(files[0:10])
        print(f"Found {len(files)} audio file candidates")

        # Filter files
        self.infos = []
        for f in files:
            dur = None
            zipfile_idx = None

            if f in zipped_files:
                zipfile_idx = zipped_to_archive[f]

            # if self.check_files:
            try:
                metadata_path = f.parent.joinpath(f.stem).with_suffix('.metadata')
                # joj = str(info_file_path)
                if zipfile_idx is not None:
                    zf = self.zipfiles[zipfile_idx]
                    with zf.open(str(f)) as audiofile:
                        info = torchaudio.info(io.BytesIO(audiofile.read()), False)
                else:
                    if metadata_path.is_file():
                        with open(metadata_path, 'r') as metadata_file:
                            info = torch_metadata_from_dict(json.load(metadata_file))
                    else:
                        if f.suffix != '.m4a':
                            info = torchaudio.info(f)
                        else:
                            pydub_info = mediainfo(f)
                            info = torchaudio.backend.common.AudioMetaData(
                                sample_rate = int(pydub_info['sample_rate']),
                                num_frames = int(pydub_info['duration_ts']),
                                num_channels = int(pydub_info['channels']),
                                bits_per_sample = int(pydub_info['bits_per_sample']),
                                encoding = 'UNKNOWN',
                            )
                        if self.create_metadata_files:
                            with open(metadata_path, 'w') as metadata_file:
                                json.dump(info.__dict__, metadata_file)
            except Exception as e:
                continue
            if info is None:
                continue
            print(info)

            if self.unit == 'seconds':
                dur = info.num_frames / info.sample_rate
            elif self.unit == 'frames':
                # See torchaudio.functional._apply_sinc_resample_kernel
                dur = int(math.ceil(self.sr * info.num_frames / info.sample_rate))

            if dur < self.min_duration or dur > self.max_duration:
                continue

            info = AudioFileInfo(
                path = f,
                metadata = info,
                duration = dur,
                zipfile_idx = zipfile_idx
            )
            print(info)

            if self.filter_fn is not None and not self.filter_fn(info):
                continue

            # finally add
            durs.append(dur)
            self.infos.append(info)

            # import json
            # lol = json.dump(info.metadata)
            # print("lol")
        
        self.cumsum_durs = np.cumsum(np.array(durs))
        print(f"Keeping {len(self.infos)} files")
        if len(self.cumsum_durs) > 0:
            total_dur = self.cumsum_durs[-1]
            print(f"{total_dur if self.unit == 'seconds' else total_dur / self.sr} secs of audio in total")
        else:
            print("No valid audio files found.")

    def get_index_offset(self, item):
        if self.sample_len is None:
            return item, 0
        else:
            # half_interval = int(self.sample_len / 2)
            half_interval = self.sample_len / 2
            shift = (
                ((np.random.random() * self.sample_len) - half_interval)
                if self.aug_shift else 
                0
            )
            # shift = np.random.randint(-half_interval, half_interval) if self.aug_shift else 0

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

    def load_audio(self, item, offset=0, duration=None):
        # Get info
        info = self.infos[item]
        sr = info.metadata.sample_rate
        curr_format = None
        duration = duration or info.duration

        # to seconds
        if self.unit == 'frames':
            offset = offset / self.sr
            duration = duration / self.sr

        # Load audio file
        try:
            if info.zipfile_idx is not None:
                zf = self.zipfiles[info.zipfile_idx]
                with self._lock:
                    with zf.open(info.filename) as songfile:
                        # audio, sr = soundfile.read(io.BytesIO(songfile.read()))
                        tmp = io.BytesIO(songfile.read())
                        audio, sr = librosa.load(
                            tmp,
                            sr = self.sr,
                            offset = offset,
                            duration = duration,
                            mono = self.mono,
                        )
            else:
                t0 = time.time()
                if info.path.suffix == '.mp3':
                    curr_format = self.format if self.format in ('f32', 's16') else 'f32'
                    audio, sr = read_mp3_file(
                        str(info.path), 
                        seek_frame = int(offset * sr), 
                        frames_to_read = int(duration * sr),
                        # format = curr_format
                    )
                    audio = torch.from_numpy(audio)
                elif info.path.suffix == '.m4a':
                    curr_format = 'f32'
                    seg = AudioSegment.from_file(
                        info.path,
                        start_second = offset,
                        duration = duration
                    )
                    audio, sr = pydub_to_np(seg)
                    audio = np.transpose(audio, (1,0))
                    audio = torch.from_numpy(audio)
                else:
                    curr_format = 'f32'
                    audio, sr = torchaudio.load(
                        info.path,
                        frame_offset = int(offset * sr),
                        num_frames = int(duration * sr),
                    )
                if self._debug is not None:
                    self._debug['audio_load_total_dur'] += (time.time() - t0)

            # To mono-channel (optional)
            if self.mono:
                t0 = time.time()
                if len(audio.shape) == 2 and audio.shape[0] == 2:
                    audio = audio.mean(dim=0, keepdims=True)
                elif len(audio.shape) == 1:
                    audio = audio[None, :]
                if self._debug is not None:
                    with self._lock:
                        self._debug['audio_mono_total_dur'] = self._debug['audio_mono_total_dur'] + (time.time() - t0)

            # Resample (optional)
            if self.sr is not None and self.sr > 0 and self.sr != sr:
                t0 = time.time()
                resampler = Resample(sr, self.sr)
                audio = resampler(audio).to(audio.dtype)
                sr = self.sr
                if self._debug is not None:
                    with self._lock:
                        self._debug['audio_resample_total_dur'] = self._debug['audio_resample_total_dur'] + (time.time() - t0)
            
            # Center (optional)
            if self.center:
                t0 = time.time()
                audio = audio - audio.mean()
                if self._debug is not None:
                    with self._lock:
                        self._debug['audio_center_total_dur'] = self._debug['audio_center_total_dur'] + (time.time() - t0)

            # Bring to correct format
            if self.format == 's16' and curr_format == 'f32':
                audio = (audio * (1 << 15)).to(torch.int16)
            elif self.format in ('f32', 'mel', 'cqt') and curr_format == 's16':
                audio = (audio.to(torch.float32) / (1 << 15))

            if self.to_spec:
                t0 = time.time()
                audio = self.compute_fbanks(audio, sr)
                if self._debug is not None:
                    with self._lock:
                        self._debug['spec_calc_total_dur'] = self._debug['spec_calc_total_dur'] + (time.time() - t0)

            # Optional cutting
            if (self.least_sl_div is not None) and (audio.shape[1] % self.least_sl_div != 0):
                new_len = (audio.shape[1] // self.least_sl_div) * self.least_sl_div
                audio = audio[:, :new_len]

            audio = audio.numpy()

        except Exception as e:
            print(f"WARNING: {e}. skipping corrupt audio file.")
            audio, sr = None, None

        # assert audio.shape[1] == self.sample_len
        
        return audio, sr

    def set_return_info(self, ret_info):
        self.ret_info = ret_info

    def __len__(self):
        if self.sample_len is None:
            return len(self.infos)
        else:
            return int(np.floor(self.cumsum_durs[-1] / self.sample_len))

    def __getitem__(self, item):
        if item >= len(self): raise IndexError
        idx, offset = self.get_index_offset(item)
        audio, sr = self.load_audio(idx, offset, self.sample_len)
        if not self.ret_info:
            return audio
        else:
            info = self.infos[idx]
            return (
                audio, 
                LoadedAudioFileInfo(
                    path = info.path,
                    metadata = info.metadata,
                    duration = info.duration,
                    zipfile_idx = info.zipfile_idx,
                    snippet_offset = offset if self.unit == 'seconds' else offset / self.sr,
                    snippet_duration = self.sample_len if self.unit == 'seconds' else self.sample_len / self.sr
                )
            )

    def __del__(self):
        for zf in self.zipfiles:
            zf.close()

    def get_collate_fn(self, pad=False, to_torch=True):
        def collate_fn(batch):
            if self.ret_info:
                audios, infos = [], []
                for b in batch:
                    if b[0] is not None and b[1] is not None:
                        audios.append(b[0])
                        infos.append(b[1])
            else:
                audios = [b for b in batch if b is not None]
            
            if pad:
                # max_len = max([a.shape[1] for a in audios])
                max_len = self.sample_len
                audios = [
                    np.pad(
                        a, 
                        ((0,0), (0, max_len - a.shape[1])), 
                        'constant', 
                        constant_values = ((0, 0), (0, 0))
                    ) 
                    for a in audios
                ]
            else:
                min_len = min([a.shape[1] for a in audios])
                audios = [a[:, :min_len] for a in audios]
            
            audios = np.stack(audios)

            if to_torch:
                audios = torch.from_numpy(audios)

            if not self.ret_info:
                return audios
            else:
                return audios, infos

        return collate_fn

    def compute_fbanks(self, audio, sr):
        if self.spec_type == "cqt":
            if len(audio.shape) == 2:
                audio = audio[0]
            spec = torch.from_numpy(librosa.cqt(
                audio.numpy(),
                sr = sr,
                fmin = self.spec_fmin,
                n_bins = self.spec_mel_bins,
                bins_per_octave = self.spec_bins_per_octave,
                hop_length = self.spec_frame_shift
            )).transpose(0,1)
            spec = spec.abs()
            if self.spec_use_power:
                spec.pow_(2)
            spec = torch.max(spec, torch.tensor(torch.finfo(torch.float).eps)).log()

        elif self.spec_type == "mel":
            if len(audio.shape) == 1:
                audio = audio[None, :]

            # features.MelSpectrogram(
            #     sr=self.sr,
            #     n_fft=
            # )
            
            spec = kaldi.fbank(
                audio,
                htk_compat=True,
                sample_frequency=sr,
                use_energy=False,
                window_type='hanning',
                num_mel_bins=self.spec_mel_bins,
                dither=0.0,
                frame_shift=(self.spec_frame_shift / sr) * 1000,
                frame_length=self.spec_frame_length,
                use_power = self.spec_use_power
            )
        # import matplotlib.pyplot as plt
        # plt.matshow(spec.to(torch.float16).float().repeat_interleave(int(spec.shape[0] / spec.shape[1]), dim=1).transpose(0,1))
        # plt.show()
        return spec

    def avg_sample_secs(self):
        if self.sample_len is not None:
            l = self.sample_len
        else:
            l = self.cumsum_durs[-1] / len(self.cumsum_durs)
        if self.unit == 'frames':
            l /= self.sr
        return l

    
def get_audio_info(file):
    return torchaudio.info(file)

def pydub_to_np(audio: pydub.AudioSegment) -> Tuple[np.ndarray, int]:
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0]. 
    Returns tuple (audio_np_array, sample_rate).
    """
    return (
        np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels)) / (1 << (8 * audio.sample_width - 1)), 
        audio.frame_rate
    )