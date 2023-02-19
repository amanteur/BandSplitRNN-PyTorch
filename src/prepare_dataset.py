from data import SAD
import musdb
import torch
from omegaconf import OmegaConf
from pathlib import Path

from typing import Iterable, Optional, List
from tqdm import tqdm


def prepare_save_line(
        track_name: str,
        start_indices: torch.Tensor,
        window_size: int
) -> Iterable[str]:
    """
    Creates string in format TRACK_NAME START_INDEX END_INDEX.
    """
    for i in start_indices:
        save_line = f"{track_name}\t{i}\t{i + window_size}\n"
        yield save_line


def run_program(
        file_path: Path,
        target: str,
        db: musdb.DB,
        sad: SAD,
):
    """
    Saves track's name and fragments indices to provided .txt file.
    """
    with open(file_path, 'w') as wf:
        for track in tqdm(db):
            # get audio data and transform to torch.Tensor
            y = torch.tensor(
                track.targets[target].audio.T,
                dtype=torch.float32
            )
            # find indices of salient segments
            indices = sad.calculate_salient_indices(y)
            # write to file
            for line in prepare_save_line(track.name, indices, sad.window_size):
                wf.write(line)
    return None


def main(
        db_dir: str,
        save_dir: str,
        subset: str,
        split: Optional[str],
        targets: List[str],
        cfg_sad: OmegaConf
):
    # initialize MUSDB parser
    db = musdb.DB(
        root=db_dir,
        subsets=subset,
        split=split,
        download=False,
        is_wav=True,
    )
    # initialize Source Activity Detector
    sad = SAD(**cfg_sad)

    # initialize directories where to save indices
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    for target in targets:
        if subset == split == 'train':
            file_path = save_dir / f"{target}_train.txt"
        elif subset == 'train' and split == 'valid':
            file_path = save_dir / f"{target}_valid.txt"
        else:
            file_path = save_dir / f"{target}_test.txt"
        # segment data and save indices to .txt file
        run_program(file_path, target, db, sad)

    return None


if __name__ == '__main__':
    # TODO: Add argparse
    db_dir = '../../../datasets/musdb18hq'
    save_dir = '../src/files/'
    subset = 'train'  # 'test'
    split = 'valid'  # 'valid'
    targets = ['vocals']
    cfg_sad = OmegaConf.load('./conf/sad/default.yaml')


    main(
        db_dir,
        save_dir,
        subset,
        split,
        targets,
        cfg_sad,
    )
