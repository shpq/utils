import os
import warnings

from omegaconf import DictConfig
import hydra
import shutil

from code_src.train import train_model
from code_src.load_pictures import extend_original_pics_storage


def save_code():
    print(hydra.utils.get_original_cwd())
    print(os.getcwd())
    os.makedirs(os.path.join(os.getcwd(), "code_src"), exist_ok=True)
    code_files_dir = os.path.join(hydra.utils.get_original_cwd(), "code_src")
    # add all train files / move them to directory
    for filename in os.listdir(code_files_dir):
        if ".py" not in filename:
            continue

        shutil.copy2(
            os.path.join(code_files_dir, filename),
            os.path.join(os.getcwd(), "code_src", filename),
        )


warnings.filterwarnings("ignore")


def os_based_path(path):
    if path:
        return os.path.join(hydra.utils.get_original_cwd(), os.path.join(*path.replace("\\", "/").split("/")))
    return ""


@hydra.main(config_path='conf', config_name='config')
def run(cfg: DictConfig) -> None:
    print(cfg.pretty())
    if cfg.general.save_code:
        save_code()
    cfg.dataset.path = os_based_path(cfg.dataset.path)
    cfg.dataset.csv_path = os_based_path(cfg.dataset.csv_path)
    cfg.dataset.csvs_path = os_based_path(cfg.dataset.csvs_path)
    cfg.dataset.csv_path_back = os_based_path(cfg.dataset.csv_path_back)
    cfg.model.pretrained_path = os_based_path(cfg.model.pretrained_path)
    extend_original_pics_storage(cfg)
    train_model(cfg)


if __name__ == '__main__':
    run()
