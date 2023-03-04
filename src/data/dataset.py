import torch
from torch.utils.data import Dataset
import torchaudio
import torch.nn.functional as F

from pathlib import Path
from typing import List, Tuple, Union


class SourceSeparationDataset(Dataset):
    """
    Dataset class for working with train/validation data from MUSDB18 dataset.
    """

    def __init__(
            self,
            file_dir: str,
            txt_dir: str = None,
            txt_path: str = None,
            target: str = 'vocals',
            is_mono: bool = False,
            mode: str = 'train',  # valid
            sr: int = 44100
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

    def get_filelist(self) -> List[Tuple[Path, Tuple[int, int]]]:
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
                (filepath_template, (int(start_idx), int(end_idx)))
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

    def __getitem__(
            self,
            index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Each Tensor's output shape: [n_channels, frames_in_segment]
        """
        fp_template, indices = self.filelist[index]
        mix_segment = self.load_file(
            str(fp_template).format('mixture'), indices
        )
        target_segment = self.load_file(
            str(fp_template).format(self.target), indices
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

        # initialized while testing
        self.eval_step = None


    def get_filelist(self) -> List[Path]:
        filelist = []
        test_dir = self.file_dir / self.mode
        for fp in test_dir.glob('*'):
            filepath_template = fp / "{}.wav"
            filelist.append(filepath_template)
        return filelist

    def pad(self, y: torch.Tensor, sr: int):
        """
        Pads audio in order to preserve all fragments while chunking.
        """
        C, T = y.shape

        # padding for preserving all chunks
        pad_size2 = int((T + self.pad_size * 2) / self.hop_size) * self.hop_size + self.win_size - T - self.pad_size * 2
        y = F.pad(y, (self.pad_size, self.pad_size + pad_size2), 'constant')
        return y

    def load_file(
            self, file_path: str, is_tgt: bool = False
    ) -> Tuple[torch.Tensor, int]:
        assert Path(file_path).is_file(), f"There is no such file - {file_path}."
        y, sr = torchaudio.load(
            file_path,
            channels_first=True
        )
        orig_length = y.shape[-1]
        # add padding
        y = self.pad(y, sr)
        # setting to mono if necessary
        if self.is_mono:
            y = torch.mean(y, dim=0, keepdim=True)
        # chunking if it is a mix
        if not is_tgt:
            y = y.unfold(
                -1, self.win_size, self.hop_size
            )
            y = y.transpose(0, 1)
        return y, orig_length

    def __getitem__(
            self, index: int
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, float], int]:
        fp_template = self.filelist[index]
        y, orig_length = self.load_file(
            str(fp_template).format('mixture'), is_tgt=False
        )
        if self.target:
            y_tgt, _ = self.load_file(
                str(fp_template).format(self.target), is_tgt=True
            )
        else:
            y_tgt = torch.Tensor(float('nan'))
        return (
            y, y_tgt, orig_length
        )

    def __len__(self):
        return len(self.filelist)
