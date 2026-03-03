from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
import os
import glob
import shutil


hf_tknzr = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")


def get_c4_data(datasets_dir, args, num_proc=40):

    if "INPUT_PATH" in os.environ:
        C4_DATA_PATH = os.environ["INPUT_PATH"]
        use_input_path = True
        print(f"Using INPUT_PATH environment variable: {C4_DATA_PATH}")
    elif args.local_data and args.local_data_path:
        C4_DATA_PATH = args.local_data_path
        use_input_path = False
        print(f"Using local C4 data path from args: {C4_DATA_PATH}")
    else:
        C4_DATA_PATH = os.path.join(datasets_dir, "c4/")
        use_input_path = False
        print(f"Using default C4 data path: {C4_DATA_PATH}")

    # Define output paths
    train_bin_path = os.path.join(C4_DATA_PATH, "train.bin")
    val_bin_path = os.path.join(C4_DATA_PATH, "val.bin")

    # ─── 2. Check Cache (Skip if .bin files already exist) ───────────────────
    if os.path.exists(train_bin_path) and os.path.exists(val_bin_path):
        print(f"Found existing .bin files in {C4_DATA_PATH}, skipping preprocessing.")
        return {
            "train": train_bin_path,
            "val": val_bin_path,
        }

    # ─── 3. Prepare Directory & Discover Files ───────────────────────────────
    os.makedirs(C4_DATA_PATH, exist_ok=True)

    if use_input_path:
        # INPUT_PATH mode: Grab ALL files (handles aliases like 0_data, 1_data)
        all_files = sorted([
            os.path.join(C4_DATA_PATH, f) 
            for f in os.listdir(C4_DATA_PATH) 
            if os.path.isfile(os.path.join(C4_DATA_PATH, f)) and not f.startswith('.')
        ])
        if not all_files:
            raise ValueError(f"No files found in INPUT_PATH: {C4_DATA_PATH}")
        data_files_list = all_files
        print(f"Found {len(data_files_list)} files in INPUT_PATH (aliases handled).")
    else:
        # Standard mode: Expect specific C4 naming convention
        data_files_list = glob.glob(os.path.join(C4_DATA_PATH, "c4-train.*.json.gz"))
        if not data_files_list:
            raise ValueError(f"No C4 .json.gz files found in {C4_DATA_PATH}.")
        print(f"Found {len(data_files_list)} standard C4 shards.")

    # ─── 4. Load Dataset ─────────────────────────────────────────────────────
    print("Loading dataset from files...")
    dataset = load_dataset("json", data_files={"train": data_files_list})

    # ─── 5. Split Train/Val ──────────────────────────────────────────────────
    print("Splitting into train/val...")
    split_dataset = dataset["train"].train_test_split(
        test_size=0.0005, seed=getattr(args, 'data_seed', 2357), shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")

    # ─── 6. Tokenize ─────────────────────────────────────────────────────────
    def process(example):
        ids = hf_tknzr.encode(
            text=example["text"],
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        out = {"ids": ids, "len": len(ids)}
        return out

    print(f"Tokenizing dataset (num_proc={num_proc})...")
    tokenized = split_dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # ─── 7. Write to Binary ───────────────────
    for split, dset in tokenized.items():
        filename = os.path.join(C4_DATA_PATH, f"{split}.bin")
        print(f"Writing {split} split to {filename}...")

        # Calculate required size
        arr_len = int(np.sum(dset["len"]))
        total_size_bytes = arr_len * 2  # 2 bytes for uint16
        
        # Check disk space before writing
        free_space = shutil.disk_usage(C4_DATA_PATH).free
        if free_space < total_size_bytes * 1.1:  # 10% buffer
            raise MemoryError(
                f"Insufficient disk space! Need {total_size_bytes / 1e9:.2f} GB, "
                f"have {free_space / 1e9:.2f} GB free in {C4_DATA_PATH}"
            )
        
        print(f"  - Total tokens: {arr_len:,}")
        print(f"  - File size: {total_size_bytes / 1e9:.2f} GB")
        print(f"  - Free space: {free_space / 1e9:.2f} GB")

        # Write using standard file I/O (NOT memmap - avoids SIGBUS)
        with open(filename, "wb") as f:
            total_batches = min(1024, len(dset))
            
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"]).astype(np.uint16)
                f.write(arr_batch.tobytes())

        # Verify file was written correctly
        written_size = os.path.getsize(filename)
        print(f"Finished writing {filename} ({written_size / 1e9:.2f} GB)")
        
        if written_size != total_size_bytes:
            print(f"Warning: Expected {total_size_bytes} bytes, got {written_size} bytes")

    # ─── 8. Return Paths (Compatible with DataReader) ────────────────────────
    return {
        "train": train_bin_path,
        "val": val_bin_path,
    }
    # if "INPUT_PATH" in os.environ:
    #     C4_DATA_PATH = os.environ["INPUT_PATH"]
    #     use_input_path = True
    #     print(f"Using INPUT_PATH environment variable: {C4_DATA_PATH}")
    # elif args.local_data and args.local_data_path:
    #     C4_DATA_PATH = args.local_data_path
    #     use_input_path = False
    #     print(f"Using local C4 data path from args: {C4_DATA_PATH}")
    # else:
    #     C4_DATA_PATH = os.path.join(datasets_dir, "c4/")
    #     use_input_path = False
    #     print(f"Using default C4 data path: {C4_DATA_PATH}")

    # train_bin_path = os.path.join(C4_DATA_PATH, "train.bin")
    # val_bin_path = os.path.join(C4_DATA_PATH, "val.bin")

    # if not os.path.exists(train_bin_path):
    #     os.makedirs(C4_DATA_PATH, exist_ok=True)

    #     if use_input_path:
    #         all_files = sorted([
    #             os.path.join(C4_DATA_PATH, f) 
    #             for f in os.listdir(C4_DATA_PATH) 
    #             if os.path.isfile(os.path.join(C4_DATA_PATH, f)) and not f.startswith('.')
    #         ])
    #         if not all_files:
    #             raise ValueError(f"No files found in INPUT_PATH: {C4_DATA_PATH}")
    #         data_files_list = all_files
    #         print(f"Found {len(data_files_list)} files in INPUT_PATH (aliases handled).")
    #     else:
    #         data_files_list = glob.glob(os.path.join(C4_DATA_PATH, "c4-train.*.json.gz"))
    #         if not data_files_list:
    #             raise ValueError(f"No C4 .json.gz files found in {C4_DATA_PATH}.")
    #         print(f"Found {len(data_files_list)} standard C4 shards.")

    #     dataset = load_dataset("json", data_files={"train": data_files_list})

    #     split_dataset = dataset["train"].train_test_split(
    #         test_size=0.0005, seed=2357, shuffle=True
    #     )
    #     split_dataset["val"] = split_dataset.pop("test")

    #     def process(example):
    #         ids = hf_tknzr.encode(
    #             text=example["text"],
    #             add_special_tokens=True,
    #             padding=False,
    #             truncation=False,
    #         )
    #         out = {"ids": ids, "len": len(ids)}
    #         return out

    #     # tokenize the dataset
    #     tokenized = split_dataset.map(
    #         process,
    #         remove_columns=["text"],
    #         desc="tokenizing the splits",
    #         num_proc=num_proc,
    #     )

    #     # concatenate all the ids in each dataset into one large file we can use for training
    #     for split, dset in tokenized.items():
    #         arr_len = np.sum(dset["len"])
    #         filename = os.path.join(C4_DATA_PATH, f"{split}.bin")
    #         dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
    #         arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
    #         total_batches = min(1024, len(dset))

    #         idx = 0
    #         for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
    #             # Batch together samples for faster write
    #             batch = dset.shard(
    #                 num_shards=total_batches, index=batch_idx, contiguous=True
    #             ).with_format("numpy")
    #             arr_batch = np.concatenate(batch["ids"])
    #             # Write into mmap
    #             arr[idx : idx + len(arr_batch)] = arr_batch
    #             idx += len(arr_batch)
    #         arr.flush()

    # return {
    #     "train": os.path.join(C4_DATA_PATH, "train.bin"),
    #     "val": os.path.join(C4_DATA_PATH, "val.bin"),
    # }
