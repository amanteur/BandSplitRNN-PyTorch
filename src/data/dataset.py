import torch
from torch.utils.data import Dataset
import torchaudio
import torch.nn.functional as F
import random
from pathlib import Path
from typing import List, Set, Tuple, Union
import numpy as np


class SourceSeparationDataset(Dataset):
    """
    Dataset class for working with train/validation data from MUSDB18 dataset.
    """
    TARGETS: Set[str] = {'vocals', 'bass', 'drums', 'other'}

    def __init__(
            self,
            file_dir: str,
            txt_dir: str = None,
            txt_path: str = None,
            target: str = 'vocals',
            is_mono: bool = False,
            mode: str = 'train',  # valid
            sr: int = 44100,
            silent_prob: float = 0.1,
            mix_prob: float = 0.1,
            mix_version: str = 'v1',
            mix_tgt_too: bool = False,
            mix_dbs: Tuple[int, int] = (0, 5)
    ):
        self.file_dir = Path(file_dir)
        self.mode = mode
        self.target = target
        self.sr = sr

        if txt_path is None and txt_dir is not None:
            self.txt_path = Path(txt_dir) / f"{target}_{mode}.txt"
        elif txt_path is not None and txt_dir is None:
            self.txt_path = Path(txt_path)
        else:
            raise ValueError("You need to specify either 'txt_path' or 'txt_dir'.")

        self.is_mono = is_mono
        self.filelist = self.get_filelist()

        # augmentations
        self.silent_prob = silent_prob
        self.mix_prob = mix_prob
        self.mix_version = mix_version
        self.mix_tgt_too = mix_tgt_too
        self.mix_dbs = mix_dbs

    def get_filelist(self) -> List[Tuple[str, Tuple[int, int]]]:
        filename2label = {}
        filelist = []
        i = 0
        for line in open(self.txt_path, 'r').readlines():
            file_name, start_idx, end_idx = line.split('\t')
            if file_name not in filename2label:
                filename2label[file_name] = i
                i += 1
            filepath_template = self.file_dir / "train" / f"{file_name}" / "{}.wav"
            filelist.append(
                (str(filepath_template), (int(start_idx), int(end_idx)))
            )
        return filelist

    def load_file(
            self,
            file_path: str,
            indices: Tuple[int, int]
    ) -> torch.Tensor:
        assert Path(file_path).is_file(), f"There is no such file - {file_path}."

        offset = indices[0]
        num_frames = indices[1] - indices[0]
        y, sr = torchaudio.load(
            file_path,
            frame_offset=offset,
            num_frames=num_frames,
            channels_first=True
        )
        assert sr == self.sr, f"Sampling rate should be equal {self.sr}, not {sr}."
        if self.is_mono:
            y = torch.mean(y, dim=0, keepdim=True)
        return y

    def load_files(
            self, fp_template: str, indices: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mix_segment = self.load_file(
            fp_template.format('mixture'), indices
        )
        tgt_segment = self.load_file(
            fp_template.format(self.target), indices
        )
        max_norm = max(
            mix_segment.abs().max(), tgt_segment.abs().max()
        )
        mix_segment /= max_norm
        tgt_segment /= max_norm
        return (
            mix_segment, tgt_segment
        )

    @staticmethod
    def imitate_silent_segments(
            mix_segment: torch.Tensor,
            tgt_segment: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            mix_segment - tgt_segment,
            torch.zeros_like(tgt_segment)
        )

    def mix_segments(
            self,
            tgt_segment: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creating new mixture and new target from target file and random multiple sources
        """
        # decide how many sources to mix
        if not self.mix_tgt_too:
            self.TARGETS.discard(self.target)
        n_sources = random.randrange(1, len(self.TARGETS) + 1)
        # decide which sources to mix
        targets_to_add = random.sample(
            self.TARGETS, n_sources
        )
        # create new mix segment
        mix_segment = tgt_segment.clone()
        for target in targets_to_add:
            # get random file to mix source from
            fp_template_to_add, indices_to_add = random.choice(self.filelist)
            segment_to_add = self.load_file(
                fp_template_to_add.format(target), indices_to_add
            )
            # normalize it
            # segment_to_add /= segment_to_add.abs().max()
            mix_segment += segment_to_add
            if target == self.target:
                tgt_segment += segment_to_add
        return (
            mix_segment, tgt_segment
        )

    def __getitem__(
            self,
            index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Each Tensor's output shape: [n_channels, frames_in_segment]
        """
        # load file templates
        fp_template, indices = self.filelist[index]

        # load files
        mix_segment, target_segment = self.load_files(fp_template, indices)

        # augmentations related to mixing/dropping sources
        if self.mode == 'train':
            # dropping target
            if random.random() < self.silent_prob:
                mix_segment, target_segment = self.imitate_silent_segments(
                    mix_segment, target_segment
                )
            # mixing with other sources
            if random.random() < self.mix_prob:
                mix_segment, target_segment = self.mix_segmentsV2(
                    target_segment
                )

        return (
            mix_segment, target_segment
        )

    def __len__(self):
        return len(self.filelist)


class TestSourceSeparationDataset(Dataset):
    """
    Dataset class for working with test data from MUSDB18 dataset.
    """

    def __init__(
            self,
            file_dir: str,
            target: str = 'vocals',
            is_mono: bool = False,
            sr: int = 44100,
            win_size: float = 3,
            hop_size: float = 0.5,
    ):
        self.file_dir = Path(file_dir)
        self.target = target
        self.is_mono = is_mono
        self.mode = 'test'
        self.sr = sr
        self.win_size = int(win_size * sr)
        self.hop_size = int(hop_size * sr)
        self.pad_size = self.win_size - self.hop_size

        self.filelist = self.get_filelist()

    def get_filelist(self) -> List[Path]:
        filelist = []
        test_dir = self.file_dir / self.mode
        for fp in test_dir.glob('*'):
            filepath_template = fp / "{}.wav"
            filelist.append(filepath_template)
        return filelist

    def pad(self, y: torch.Tensor):
        """
        Pads audio in order to preserve all fragments while chunking.
        """

        # padding for preserving all chunks
        # C, T = y.shape
        # pad_size2 = self.win_size - (T + self.pad_size * 2) % self.hop_size
        # y = F.pad(y, (self.pad_size, self.pad_size + pad_size2), 'constant')

        # w/o additional padding
        y = F.pad(y, (self.pad_size, self.pad_size), 'constant')
        return y

    def load_file(
            self, file_path: str, is_tgt: bool = False
    ) -> Tuple[torch.Tensor, int]:
        assert Path(file_path).is_file(), f"There is no such file - {file_path}."
        y, sr = torchaudio.load(
            file_path,
            channels_first=True
        )
        if sr != self.sr:
            y = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sr
            )(y)
        # add padding
        y = self.pad(y)
        duration = y.shape[-1]
        # setting to mono if necessary
        if self.is_mono:
            y = torch.mean(y, dim=0, keepdim=True)
        # chunking if it is a mix
        if not is_tgt:
            y = y.unfold(
                -1, self.win_size, self.hop_size
            )
            y = y.transpose(0, 1)
        return y, duration

    def __getitem__(
            self, index: int
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, float], int]:
        fp_template = self.filelist[index]
        y, duration = self.load_file(
            str(fp_template).format('mixture'), is_tgt=False
        )
        if self.target:
            y_tgt, _ = self.load_file(
                str(fp_template).format(self.target), is_tgt=True
            )
        else:
            y_tgt = torch.Tensor(float('nan'))
        return (
            y, y_tgt, duration
        )

    def __len__(self):
        return len(self.filelist)


class InferenceSourceSeparationDataset(TestSourceSeparationDataset):
    def __init__(
            self,
            in_file_path: str,
            out_file_path: str,
            target: str,
            is_mono: bool = False,
            sr: int = 44100,
            win_size: float = 3,
            hop_size: float = 0.5
    ):
        self.in_file_path = Path(in_file_path)
        self.out_file_path = Path(out_file_path)

        self.target = target
        self.is_mono = is_mono
        self.sr = sr
        self.win_size = int(win_size * sr)
        self.hop_size = int(hop_size * sr)
        self.pad_size = self.win_size - self.hop_size

        self.filelist = self.get_filelist()

    def get_filelist(self) -> List[Tuple[Path, Path]]:
        extensions = ['.wav', '.mp3']
        filelist = []
        if self.in_file_path.is_file() and self.in_file_path.suffix in extensions:
            self.out_file_path = self.out_file_path / f"{self.in_file_path.stem}_{self.target}.wav"
            filelist.append((self.in_file_path, self.out_file_path))
        elif self.in_file_path.is_dir():
            for in_fp in self.in_file_path.glob("*"):
                if in_fp.suffix in extensions:
                    out_fp = self.out_file_path / f"{in_fp.stem}_{self.target}.wav"
                    filelist.append((in_fp, out_fp))
        else:
            raise ValueError(f"Can not open the path {self.in_file_path}")
        return filelist

    def __getitem__(
            self, index: int
    ) -> Tuple[torch.Tensor, int, Path]:
        in_fp, out_fp = self.filelist[index]
        y, duration = self.load_file(str(in_fp), is_tgt=False)
        return y, duration, out_fp
