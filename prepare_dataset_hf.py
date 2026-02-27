import os
import pandas as pd
from huggingface_hub import hf_hub_download
from datasets import Dataset, DatasetDict
from normalize_text import normalize_aviation_text

def load_and_normalize():
    print("Downloading ATC ASR Dataset Parquet files directly...")
    
    repo_id = "jacktol/ATC-ASR-Dataset"
    repo_type = "dataset"
    
    # Список всех файлов из репозитория
    files_to_download = {
        "train": [
            "data/train-00000-of-00002.parquet",
            "data/train-00001-of-00002.parquet"
        ],
        "validation": [
            "data/validation-00000-of-00001.parquet"
        ],
        "test": [
            "data/test-00000-of-00001.parquet"
        ]
    }

    local_audio_dir = os.path.join(os.getcwd(), "audio_files")
    os.makedirs(local_audio_dir, exist_ok=True)
    print(f"Audio files will be extracted to: {local_audio_dir}")

    new_splits = {}
    text_column = "text"

    for split_name, file_paths in files_to_download.items():
        print(f"\nProcessing split: {split_name}")
        
        all_dfs = []
        for file_path in file_paths:
            try:
                local_parquet_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    repo_type=repo_type
                )
                print(f"Downloaded {file_path}")
                df_part = pd.read_parquet(local_parquet_path)
                all_dfs.append(df_part)
            except Exception as e:
                print(f"Failed to download {file_path}. Error: {e}")

        if not all_dfs:
            print(f"Skipping {split_name} (no files loaded).")
            continue
            
        # Склеиваем все куски для текущего сплита
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"Loaded total {len(df)} rows for {split_name}.")

        audio_paths = []
        texts = []

        for index, row in df.iterrows():
            audio_data = row['audio']
            
            if not isinstance(audio_data, dict) or 'bytes' not in audio_data:
                continue
                
            audio_bytes = audio_data['bytes']
            original_path = audio_data.get('path', f"audio_{index}.wav")
            
            # Если пути нет или он пустой, генерируем простое имя
            if not original_path:
                original_path = f"audio_{index}.wav"
                
            safe_filename = os.path.basename(original_path)
            filename = f"{split_name}_{index}_{safe_filename}"
            dst_path = os.path.join(local_audio_dir, filename)

            # Сохраняем файл на диск (только если его еще нет)
            if not os.path.exists(dst_path):
                with open(dst_path, "wb") as f:
                    f.write(audio_bytes)

            audio_paths.append(dst_path)
            texts.append(normalize_aviation_text(row[text_column]))

            if index > 0 and index % 1000 == 0:
                print(f"  Extracted {index}/{len(df)} files...")

        new_splits[split_name] = Dataset.from_dict({
            "audio_path": audio_paths,
            text_column: texts,
        })
        print(f"Successfully processed {split_name}: {len(audio_paths)} examples.")

    final_dataset = DatasetDict(new_splits)
    print("\nFinished extraction!")
    print(final_dataset)
    return final_dataset, text_column

if __name__ == "__main__":
    ds, col = load_and_normalize()
