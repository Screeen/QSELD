import os
from utils import list_to_string

def make_list(x):
    return x if isinstance(x, list) else [x]


def create_symlink_dir(base_folder, label_or_spec='label', overlaps=1, splits=1):

    ovs = make_list(overlaps)
    splits = make_list(splits)
    label_dirs = []
    suffix = ''

    for ov in ovs:
        for split in splits:
            suffix = '_nfft512_regr0' if label_or_spec == 'label' else '_30db_nfft512_norm'
            dir_name = f"{label_or_spec}_ov{ov}_split{split}" + suffix
            label_dirs.append(dir_name)

    label_dirs_filtered = []
    for file_cnt, dir_name in enumerate(os.listdir(base_folder)):
        if dir_name in label_dirs:
            label_dirs_filtered.append(dir_name)
            print(f"dir_name {dir_name}")

    symlink_dir_path = None
    if label_dirs_filtered:
        symlink_dir_name = f"{label_or_spec}_ov{list_to_string(ovs)}_split{list_to_string(splits)}" + suffix
        symlink_dir_path = os.path.join(base_folder, symlink_dir_name)
        os.makedirs(symlink_dir_path, exist_ok=True)
        print(f"creating {symlink_dir_path}")
    else:
        print(f"os.listdir(base_folder) \n{os.listdir(base_folder)}")
        print(f"label_dirs \n{label_dirs}")
        raise ValueError(f"label_dirs_filtered \n{label_dirs_filtered}")

    if not symlink_dir_path:
        raise FileNotFoundError

    for label_dir_filtered in label_dirs_filtered:
        abs_path_filtered = os.path.abspath(os.path.join(base_folder, label_dir_filtered))
        for file_name in os.listdir(abs_path_filtered):
            src = os.path.abspath(os.path.join(base_folder, label_dir_filtered, file_name))
            dst = os.path.abspath(os.path.join(symlink_dir_path, file_name))
            os.symlink(src, dst)
            print(src)
            print(dst)


datasets_dir = '../datasets'
dataset = 'resim'
base_folder_ = os.path.join(datasets_dir, dataset)

for overlaps in [[1]]:
    for splits in [[1, 2, 3]]:
        for type in ['label', 'spec']:
            create_symlink_dir(base_folder_, type, overlaps, splits)
