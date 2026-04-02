import os
import zipfile


def extract_zip(zip_path: str, out_dir: str) -> None:
    """解压 zip 到 out_dir（若已解压也可以重复执行，不影响）。"""
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def prepare_dataset(data_dir: str) -> str:
    """返回数据 txt 文件路径。"""
    os.makedirs(data_dir, exist_ok=True)

    zip_path = os.path.join(data_dir, "fra-eng.zip")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"未找到 {zip_path}\n"
            f"请先把 fra-eng.zip 放到该目录（{data_dir}）下。"
        )

    print(f"[local] found zip: {zip_path}")
    extract_zip(zip_path, data_dir)
    print(f"[ok] extracted into: {data_dir}")

    # ManyThings 的压缩包里通常有 fra.txt
    txt_path = os.path.join(data_dir, "fra.txt")
    if os.path.exists(txt_path):
        return txt_path

    # 有些教程会重命名为 eng-fra.txt；这里做兜底
    alt = os.path.join(data_dir, "eng-fra.txt")
    if os.path.exists(alt):
        return alt

    # 如果都没有，就把 data_dir 下的文件列出来给你定位
    files = ", ".join(sorted(os.listdir(data_dir)))
    raise FileNotFoundError(
        f"没有找到 fra.txt 或 eng-fra.txt，请检查解压结果：{data_dir}\n"
        f"当前目录文件：{files}"
    )


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    path = prepare_dataset(data_dir)
    print(f"[ok] dataset file: {path}")
