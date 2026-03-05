import torch
import numpy as np
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
from typing import Dict, Any, Iterator
import math
import gc


class PreprocessedIterableDataset(IterableDataset):
    """
    Iterable dataset that processes HuggingFace streaming datasets on-the-fly.
    This processes data step-by-step without loading everything into memory.
    """
    
    def __init__(self, dataset, tokenizer, batch_size: int, max_length: int):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through the dataset, yielding batched and tokenized data."""
        batch_texts = []
        
        try:
            for example in self.dataset:
                # Extract text from the example
                text = example.get("text", "")
                if not text:  # Skip empty texts
                    continue
                    
                batch_texts.append(text)
                
                if len(batch_texts) >= self.batch_size:
                    # Tokenize the batch
                    batch = self.tokenizer(
                        batch_texts,
                        max_length=self.max_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    yield batch
                    batch_texts = []
            
            # Handle the last incomplete batch
            if batch_texts:
                batch = self.tokenizer(
                    batch_texts,
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length", 
                    return_tensors="pt",
                )
                yield batch
                
        except Exception as e:
            print(f"Error in dataset iteration: {e}")
            # Don't crash, just stop iteration
            return


class StreamingDataReader:
    """
    Streaming data reader that processes data step-by-step.
    Compatible with your existing DataReader interface.
    """
    
    def __init__(
        self,
        dataset,
        tokenizer,
        batch_size: int,
        max_length: int,
        seed: int = 1337,
        world_size: int = 1,
        rank: int = 0,
        num_workers: int = 8,
        is_eval: bool = False,
        eval_batches: int = 32,
        empty_cache_freq: int = 32,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.seed = seed
        self.world_size = world_size
        self.rank = rank
        self.num_workers = num_workers
        self.is_eval = is_eval
        self.eval_batches = eval_batches
        self.empty_cahce_freq = empty_cache_freq
        
        print(f"Setting up StreamingDataReader:")
        print(f"  - is_eval: {is_eval}")
        print(f"  - batch_size: {batch_size}")
        print(f"  - max_length: {max_length}")
        print(f"  - eval_batches: {eval_batches if is_eval else 'N/A'}")
        print(f"  - num_workers: {num_workers}")
        
        # Create the iterable dataset
        self.iterable_dataset = PreprocessedIterableDataset(
            dataset, tokenizer, batch_size, max_length
        )
        
        # Create the dataloader
        self.dataloader = torch.utils.data.DataLoader(
            self.iterable_dataset,
            batch_size=None,  # Batching is handled by the iterable dataset
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,
        )
        
        # For compatibility with existing code
        self.step = 0
        
        # Set reasonable values for streaming datasets
        if is_eval:
            # For evaluation, we know how many batches we want
            self.num_tokens = eval_batches * batch_size * max_length
            self._num_batches = eval_batches
            print(f"Eval mode: will process {self._num_batches} batches")
        else:
            # For training, set a large but finite number
            estimated_tokens = 100_000_000  # 100M tokens
            self.num_tokens = estimated_tokens
            self._num_batches = estimated_tokens // (batch_size * max_length)
            print(f"Train mode: estimated {self._num_batches} batches available")
        
        # Iterator for the dataloader
        self._dataloader_iter = None
    
    def _get_dataloader_iter(self):
        """Get or create dataloader iterator."""
        if self._dataloader_iter is None:
            print("Creating new dataloader iterator...")
            self._dataloader_iter = iter(self.dataloader)
        return self._dataloader_iter
    
    def sample_batch(self):
        """Sample a batch of data. Compatible with existing DataReader interface."""
        try:
            dataloader_iter = self._get_dataloader_iter()
            batch = next(dataloader_iter)
            self.step += 1
            
            # Convert to the expected format (input_ids -> x, y)
            input_ids = batch["input_ids"]
            
            # Make sure we have the right sequence length for x and y
            if input_ids.size(1) < 2:
                # If sequence is too short, pad it
                pad_length = max(2, self.max_length)
                padded = torch.full((input_ids.size(0), pad_length), 
                                  self.tokenizer.pad_token_id, 
                                  dtype=input_ids.dtype)
                padded[:, :input_ids.size(1)] = input_ids
                input_ids = padded
            
            x = input_ids[:, :-1].contiguous()
            y = input_ids[:, 1:].contiguous()

            if self.step % self.empty_cahce_freq == 0:
                print("CUDA cache cleanup...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return x, y
            
        except StopIteration:
            print("DataLoader exhausted, creating new iterator...")
            # Reset the iterator if we've exhausted the dataset
            self._dataloader_iter = None
            return self.sample_batch()
        except Exception as e:
            print(f"Error in sample_batch: {e}")
            # Reset and try again
            self._dataloader_iter = None
            return self.sample_batch()
    
    def set_step(self, step):
        """Set the current step. For compatibility."""
        self.step = step
    
    def num_batches(self):
        """Return estimated number of batches. For compatibility."""
        return self._num_batches
    
    def __len__(self):
        """Return the estimated number of sequences for compatibility."""
        return max(1, self.num_tokens - self.max_length - 1)
    
    def __getitem__(self, idx):
        """Not implemented for streaming datasets."""
        raise NotImplementedError("Streaming datasets don't support indexing. Use sample_batch() instead.")
