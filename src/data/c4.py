from tqdm import tqdm
import numpy as np
import datasets
import datasets.distributed
from transformers import AutoTokenizer
from .streaming_reader import StreamingDataReader
import torch.distributed as dist
import os
import glob
import shutil


import os
import glob
import datasets
import datasets.distributed
from transformers import AutoTokenizer
from .streaming_reader import StreamingDataReader
import torch.distributed as dist


def get_tokenizer(tokenizer_name="gpt2"):
    """Get the tokenizer. Make this configurable if needed."""
    if tokenizer_name == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_c4_data_streaming(args):
    """
    Get C4 data using streaming approach for much better performance.
    This processes local files step-by-step without creating .bin files.
    """
    print("Loading C4 dataset with streaming=True for better performance...")
    
    # Handle local data
    if hasattr(args, 'local_data') and args.local_data and hasattr(args, 'local_data_path') and args.local_data_path:
        print(f"Using local C4 data from: {args.local_data_path}")
        
        # Expand the path (handle ~)
        input_path = os.path.expanduser(args.local_data_path)
        
        if not os.path.exists(input_path):
            raise ValueError(f"Local data path does not exist: {input_path}")
        
        # Look for various JSON file patterns
        json_patterns = [
            "*.json.gz",
            "*.json",
            "**/*.json.gz",  # Recursive search
            "**/*.json",
        ]
        
        json_files = []
        for pattern in json_patterns:
            found_files = glob.glob(os.path.join(input_path, pattern), recursive=True)
            json_files.extend(found_files)
        
        # Remove duplicates and sort
        json_files = sorted(list(set(json_files)))
        
        if not json_files:
            raise ValueError(f"No JSON files found in local path: {input_path}")
        
        print(f"Found {len(json_files)} JSON files for streaming processing")
        print(f"Sample files: {json_files[:3]}")
        
        # Create datasets from local files with streaming
        data_files = {"train": json_files}
        
        print("Loading train dataset from local files...")
        train_dataset = datasets.load_dataset(
            "json", 
            data_files=data_files, 
            split='train', 
            streaming=True  # This is key - no .bin files created!
        )
        
        print("Creating validation dataset from local files...")
        val_dataset = datasets.load_dataset(
            "json", 
            data_files=data_files, 
            split='train', 
            streaming=True
        ).take(2000)  # Take subset for validation
        
    else:
        # Fallback to HuggingFace (but this wasn't requested)
        print("Loading C4 dataset from HuggingFace...")
        train_dataset = datasets.load_dataset(
            "allenai/c4", 
            "en", 
            split="train", 
            streaming=True
        )
        val_dataset = datasets.load_dataset(
            "allenai/c4", 
            "en", 
            split="validation", 
            streaming=True
        )
    
    # Shuffle the datasets
    print("Shuffling datasets...")
    train_dataset = train_dataset.shuffle(seed=getattr(args, 'data_seed', 1337))
    val_dataset = val_dataset.shuffle(seed=42)  # Fixed seed for validation
    
    # Handle distributed training
    world_size = 1
    rank = 0
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        print(f"Splitting dataset for distributed training: rank {rank}/{world_size}")
        # Split dataset by node for distributed training
        train_dataset = datasets.distributed.split_dataset_by_node(
            train_dataset, rank=rank, world_size=world_size
        )
        # Note: Don't split validation dataset to ensure consistent evaluation
    
    print(f"Streaming dataset setup complete: rank {rank}/{world_size}")
    
    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "world_size": world_size,
        "rank": rank,
    }


def get_c4_data(datasets_dir, args, num_proc=40):
    """
    C4 data loader that prioritizes streaming for local data.
    """
    # Always use streaming when local_data is specified
    if hasattr(args, 'local_data') and args.local_data:
        print("Local data detected - using streaming approach to avoid .bin file creation")
        return get_c4_data_streaming(args)
    
    # Check if we should use streaming
    if getattr(args, 'streaming', True):
        print("Using streaming approach for C4 data loading...")
        return get_c4_data_streaming(args)
    else:
        print("Streaming disabled - this would create .bin files (not recommended for local data)")
        # Your original .bin file creation code would go here
        # But since you want to avoid this, we'll force streaming
        print("Forcing streaming to avoid .bin file creation...")
        return get_c4_data_streaming(args)

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
