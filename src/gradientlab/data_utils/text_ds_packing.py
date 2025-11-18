from datasets import Dataset, DatasetDict


class TextDatasetPacker:
    def __init__(self, tok, bos: str, eos: str, max_seq_len: int) -> None:
        self.bos = bos
        self.eos = eos
        self.tok = tok
        self.max_seq_len = max_seq_len

    def _encode_batch(self, batch):
        texts = [self.bos + t + self.eos for t in batch["text"]]
        return {
            "input_ids": self.tok(
                texts, add_special_tokens=False, return_attention_mask=False
            )["input_ids"]
        }

    def group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.max_seq_len:
            total_length = (total_length // self.max_seq_len) * self.max_seq_len
        # Split by chunks of block_size.
        result = {
            k: [
                t[i : i + self.max_seq_len]
                for i in range(0, total_length, self.max_seq_len)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    def pack_dataset(
        self, ds: Dataset | DatasetDict, column: str, num_proc: int = 8
    ) -> Dataset | DatasetDict:
        if column not in ds.column_names:
            raise ValueError(f"Column '{column}' not found in the dataset.")

        tok_ds = ds.map(
            self._encode_batch,
            batched=True,
            remove_columns=list(ds.column_names),
            desc="Tokenization..",
            num_proc=num_proc,
            batch_size=256,
        )

        packed_ds = tok_ds.map(
            self.group_texts,
            batched=True,
            num_proc=num_proc,
            batch_size=256,
        )
        return packed_ds
