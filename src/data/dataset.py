import torch
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
from typing import List, Tuple


class SourceSeparationDataset(Dataset):
    """
    Dataset class for working with data from MUSDB18 dataset.
    """

    def __init__(
            self,
            file_dir: str,
            txt_dir: str = None,
            txt_path: str = None,
            target: str = 'vocals',
            is_mono: bool = False,
            mode: str = 'train',
    ):
        self.file_dir = Path(file_dir)
        self.mode = mode
        self.target = target

        if txt_path is None and txt_dir is not None:
            self.txt_path = Path(txt_dir) / f"{target}_{mode}.txt"
        elif txt_path is not None and txt_dir is None:
            self.txt_path = Path(txt_path)
        else:
            raise ValueError("You need to specify either 'txt_path' or 'txt_dir'.")

        self.is_mono = is_mono
        self.filelist, self.labellist = self.get_filelist()

    def get_filelist(
            self
    ) -> Tuple[List[Tuple[Path, Tuple[int, int]]], List[int]]:
        filename2label = {}
        filelist = []
        labellist = []
        i = 0
        for line in open(self.txt_path, 'r').readlines():
            file_name, start_idx, end_idx = line.split('\t')
            if file_name not in filename2label:
                filename2label[file_name] = i
                i += 1
            filepath_template = self.file_dir / f"{self.mode}" / f"{file_name}" / "{}.wav"
            filelist.append(
                (filepath_template, (int(start_idx), int(end_idx)))
            )
            labellist.append(filename2label[file_name])
        return filelist, labellist

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
