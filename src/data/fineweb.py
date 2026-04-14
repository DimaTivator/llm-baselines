from tqdm import tqdm
import numpy as np
import datasets
import datasets.distributed
from transformers import AutoTokenizer
import torch.distributed as dist
import os
import glob


hf_tknzr = AutoTokenizer.from_pretrained("gpt2")


def _find_data_files(data_dir):
    """Find all parquet/JSON files recursively under data_dir."""
    files = glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True)
    files += glob.glob(os.path.join(data_dir, "**/*.json.gz"), recursive=True)
    files += glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    files += glob.glob(os.path.join(data_dir, "**/*.jsonl"), recursive=True)
    files = sorted(set(files))
    if not files:
        raise ValueError(f"No data files found in {data_dir}")
    print(f"Found {len(files)} data files in {data_dir}")
    return files


def _detect_format(files):
    if any(f.endswith(".parquet") for f in files):
        return "parquet"
    return "json"


FINEWEB_VARIANT = "sample-100BT"


def get_fineweb_data(datasets_dir, args, num_proc=40):
    if getattr(args, 'streaming', False):
        return get_fineweb_data_streaming(datasets_dir, args)
    else:
        return get_fineweb_data_common(datasets_dir, args, num_proc)


def get_fineweb_data_streaming(datasets_dir, args):
    eval_batch_size = getattr(args, 'eval_batch_size', args.batch_size)
    eval_batches = getattr(args, 'eval_batches', 32)
    val_examples_needed = int(eval_batch_size * eval_batches)

    if os.path.isdir(datasets_dir):
        print("====== Searching for files =======")
        data_files_list = _find_data_files(datasets_dir)
        print('\n' * 2)
        print("====== Files =======")
        print(len(data_files_list))
        print('\n' * 2)
        print("======== Data Format =======")
        fmt = _detect_format(data_files_list)
        print(fmt)
        print('\n' * 2)
        data_files = {"train": data_files_list}

        # Shuffle before split so val gets random examples, not just the
        # first N in file order. take/skip are lazy and don't share state.
        print("====== Shuffling =========")
        shuffled = datasets.load_dataset(
            fmt, data_files=data_files, split='train', streaming=True
        ).shuffle(seed=2357, buffer_size=10_000)
        print('\n' * 2)

        print("======= Train and Val =========")
        val_dataset = shuffled.take(val_examples_needed)
        train_dataset = shuffled.skip(val_examples_needed)
        # print(f"Len Val: {len(val_dataset)}")
        # print(f"Len Train: {len(train_dataset)}")
        print('\n' * 2)

        # Heuristic: ~4 bytes per token for English text in raw JSON/parquet
        estimated_tokens = sum(os.path.getsize(f) for f in data_files_list) // 4
        print(f"Estimated tokens: {estimated_tokens}")
    else:
        print(f"{datasets_dir} not found locally, streaming FineWeb-Edu ({FINEWEB_VARIANT}) from HuggingFace...")
        shuffled = datasets.load_dataset(
            "HuggingFaceFW/fineweb-edu", FINEWEB_VARIANT, split="train", streaming=True
        ).shuffle(seed=2357, buffer_size=10_000)

        val_dataset = shuffled.take(val_examples_needed)
        train_dataset = shuffled.skip(val_examples_needed)

        estimated_tokens = 100_000_000_000  # FineWeb-Edu sample-100BT ~100B tokens

    train_dataset = train_dataset.shuffle(seed=getattr(args, 'data_seed', 1337))

    world_size, rank = 1, 0
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        train_dataset = datasets.distributed.split_dataset_by_node(
            train_dataset, rank=rank, world_size=world_size
        )

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "world_size": world_size,
        "rank": rank,
        "estimated_tokens": estimated_tokens,
    }


def get_fineweb_data_common(datasets_dir, args, num_proc=40):
    train_bin_path = os.path.join(datasets_dir, "train.bin")
    val_bin_path = os.path.join(datasets_dir, "val.bin")

    if not os.path.exists(train_bin_path):
        os.makedirs(datasets_dir, exist_ok=True)

        try:
            data_files_list = _find_data_files(datasets_dir)
            fmt = _detect_format(data_files_list)
        except ValueError:
            print(f"No local files found. Downloading FineWeb-Edu ({FINEWEB_VARIANT}) from HuggingFace...")
            dataset = datasets.load_dataset("HuggingFaceFW/fineweb-edu", FINEWEB_VARIANT)
            split_dataset = dataset["train"].train_test_split(
                test_size=0.0005, seed=2357, shuffle=True
            )
            split_dataset["val"] = split_dataset.pop("test")

            def process(example):
                ids = hf_tknzr.encode(
                    text=example["text"], add_special_tokens=True,
                    padding=False, truncation=False,
                )
                return {"ids": ids, "len": len(ids)}

            tokenized = split_dataset.map(
                process, remove_columns=["text"],
                desc="tokenizing the splits", num_proc=num_proc,
            )

            for split, dset in tokenized.items():
                arr_len = np.sum(dset["len"])
                filename = os.path.join(datasets_dir, f"{split}.bin")
                dtype = np.uint16
                arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
                total_batches = min(1024, len(dset))

                idx = 0
                for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                    batch = dset.shard(
                        num_shards=total_batches, index=batch_idx, contiguous=True
                    ).with_format("numpy")
                    arr_batch = np.concatenate(batch["ids"])
                    arr[idx : idx + len(arr_batch)] = arr_batch
                    idx += len(arr_batch)
                arr.flush()

            return {"train": train_bin_path, "val": val_bin_path}

        dataset = datasets.load_dataset(fmt, data_files={"train": data_files_list})
        split_dataset = dataset["train"].train_test_split(
            test_size=0.0005, seed=2357, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")

        def process(example):
            ids = hf_tknzr.encode(
                text=example["text"], add_special_tokens=True,
                padding=False, truncation=False,
            )
            return {"ids": ids, "len": len(ids)}

        tokenized = split_dataset.map(
            process, remove_columns=["text"],
            desc="tokenizing the splits", num_proc=num_proc,
        )

        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"])
            filename = os.path.join(datasets_dir, f"{split}.bin")
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = min(1024, len(dset))

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    return {
        "train": train_bin_path,
        "val": val_bin_path,
    }
