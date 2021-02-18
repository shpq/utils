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
        if ".py" not in filename:
            continue

        shutil.copy2(
            os.path.join(code_files_dir, filename),
            os.path.join(os.getcwd(), "code_src", filename),
        )


warnings.filterwarnings("ignore")


def os_based_path(path):
    if path:
        return os.path.join(hydra.utils.get_original_cwd(), os.path.join(*path.split("/")))
    return ""


@hydra.main(config_path='conf', config_name='config')
def run(cfg: DictConfig) -> None:
    print(cfg.pretty())
    if cfg.general.save_code:
        save_code()
    cfg.dataset.path = os_based_path(cfg.dataset.path)
    cfg.dataset.csv.path = os_based_path(cfg.dataset.csv.path)
    cfg.dataset.csv.train_path = os_based_path(cfg.dataset.csv.train_path)
    cfg.dataset.csv.test_path = os_based_path(cfg.dataset.csv.test_path)
    cfg.model.path = os_based_path(cfg.model.path)
    train_model(cfg)


if __name__ == '__main__':
    run()
