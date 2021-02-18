import os
import warnings

from omegaconf import DictConfig
import hydra
import shutil

from code_src.train import train_model


def save_code():
    print(hydra.utils.get_original_cwd())
    print(os.getcwd())
    os.makedirs(os.path.join(os.getcwd(), "code_src"), exist_ok=True)
    code_files_dir = os.path.join(hydra.utils.get_original_cwd(), "code_src")
    # add all train files / move them to directory
    for filename in os.listdir(code_files_dir):
        if not filename.endswith(".py"):
            continue

        shutil.copy2(
            os.path.join(code_files_dir, filename),
            os.path.join(os.getcwd(), "code_src", filename),
        )


warnings.filterwarnings("ignore")


def os_based_path(path):
    if path is not None:
        return os.path.join(hydra.utils.get_original_cwd(), os.path.join(*path.split("/")))
    return None


@hydra.main(config_path='configs', config_name='config')
def run(cfg: DictConfig) -> None:
    print(cfg.pretty())
    save_code()
    cfg.dataset.ugly_pics = os_based_path(cfg.dataset.ugly_pics)
    cfg.dataset.beauty_pics = os_based_path(cfg.dataset.beauty_pics)
    # cfg.dataset.train_path = os_based_path(cfg.dataset.train_path)
    # cfg.dataset.test_path = os_based_path(cfg.dataset.test_path)
    cfg.model.pretrained_path = os_based_path(cfg.model.pretrained_path)
    cfg.model.beauty.path = os_based_path(cfg.model.beauty.path)
    # extend_original_pics_storage(cfg)
    train_model(cfg)


if __name__ == '__main__':
    run()
