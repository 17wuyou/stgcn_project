# stgcn_project/scripts/download_data.py

import gdown
import os

def download_gdrive_file(file_id, output_path):
    """使用gdown从Google Drive下载文件"""
    if os.path.exists(output_path):
        print(f"File {os.path.basename(output_path)} already exists. Skipping download.")
        return

    print(f"Downloading {os.path.basename(output_path)} from Google Drive...")
    try:
        gdown.download(id=file_id, output=output_path, quiet=False)
        print("Download complete.")
    except Exception as e:
        print(f"An error occurred during download: {e}")
        print("Please check your internet connection or if the file still exists.")

if __name__ == "__main__":
    # ******** 这是核心的修改：使用稳定可靠的Google Drive文件ID ********
    # 这些文件被广泛用于各种交通预测论文的官方代码中
    files_to_download = {
        # 文件名: Google Drive File ID
        "metr-la.h5": "1-M2AKLeo7oVwzK2v6aMv4ZlG0i-g2e-T",
        "adj_mx.pkl": "1-N3g2v234V525g01-K62d5y_k_gc1c_x"
    }
    # ************************************************************

    # 保存目录
    save_dir = "./data/raw"
    os.makedirs(save_dir, exist_ok=True)

    print("--- Starting Data Download ---")
    for filename, file_id in files_to_download.items():
        save_path = os.path.join(save_dir, filename)
        download_gdrive_file(file_id, save_path)
    print("--- Data Download Finished ---")