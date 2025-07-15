import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from src import get_checkpoint

# from src.chat_templates import LLAMA_3_TEMPLATE
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    apply_chat_template,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import DataCollatorForCompletionOnlyLM


class MaskedCompletionDataCollator(DataCollatorForCompletionOnlyLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def torch_call(self, examples):
        # Let parent class handle initial collation
        batch = super().torch_call(examples)

        # Retrieve token IDs for the mask markers
        mask_start_id = self.tokenizer.convert_tokens_to_ids("<MASK_START>")
        mask_end_id = self.tokenizer.convert_tokens_to_ids("<MASK_END>")

        # Get batch properties
        max_length = batch["input_ids"].size(1)  # Maximum sequence length in the batch
        pad_id = self.tokenizer.pad_token_id  # Padding token ID

        for i in range(len(batch["input_ids"])):
            input_ids = batch["input_ids"][i].tolist()

            try:
                # Locate mask tokens
                start_pos = input_ids.index(mask_start_id)
                end_pos = input_ids.index(mask_end_id)

                # Remove <MASK_START> and <MASK_END>, keeping tokens in between
                new_input_ids = input_ids[:start_pos] + input_ids[start_pos + 1 : end_pos] + input_ids[end_pos + 1 :]

                # Adjust sizes to match max_length
                if len(new_input_ids) < max_length:
                    new_input_ids += [pad_id] * (max_length - len(new_input_ids))  # Pad
                elif len(new_input_ids) > max_length:
                    new_input_ids = new_input_ids[:max_length]  # Truncate

                # Update input_ids in the batch
                batch["input_ids"][i] = torch.tensor(new_input_ids, dtype=torch.long)

                # Adjust attention mask
                attention_mask = batch["attention_mask"][i].tolist()
                new_attention_mask = (
                    attention_mask[:start_pos]
                    + attention_mask[start_pos + 1 : end_pos]
                    + attention_mask[end_pos + 1 :]
                )
                new_attention_mask = new_attention_mask[:max_length] + [0] * (
                    max_length - len(new_attention_mask)
                )  # Pad/truncate
                batch["attention_mask"][i] = torch.tensor(new_attention_mask, dtype=torch.long)

                # Adjust labels (if present)
                if "labels" in batch:
                    labels = batch["labels"][i].tolist()
                    new_labels = labels[:start_pos] + labels[start_pos + 1 : end_pos] + labels[end_pos + 1 :]
                    new_labels = new_labels[:max_length] + [-100] * (
                        max_length - len(new_labels)
                    )  # Use -100 for padding
                    batch["labels"][i] = torch.tensor(new_labels, dtype=torch.long)

            except ValueError as e:
                print(f"\nError processing example {i}:")
                print(f"Tokens in text: {set(input_ids)}")
                print(f"Decoded text: {self.tokenizer.decode(input_ids)}")
                raise ValueError(
                    f"Could not find mask tokens in example {i}. Mask start ID: {mask_start_id}, Mask end ID: {mask_end_id}"
                ) from e

        return batch


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    print(f"script_args: {script_args}")
    print(f"training_args: {training_args}")
    print(f"model_config: {model_config}")
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        print(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )

    # For base models we default to the Llama chat template
    # FIXME: base model generation is unbounded for some reason (no EOS)
    # if tokenizer.chat_template is None:
    #     tokenizer.chat_template = LLAMA_3_TEMPLATE
    #     tokenizer.eos_token = "<|eot_id|>"

    # tokenizer.pad_token = "<|end_of_text|>"

    tokenizer.add_special_tokens({"additional_special_tokens": ["<MASK_START>", "<MASK_END>"]})
    MASK_START_id = tokenizer.convert_tokens_to_ids("<MASK_START>")
    MASK_END_id = tokenizer.convert_tokens_to_ids("<MASK_END>")

    dataset = load_dataset(script_args.dataset_name)

    def process_dataset(example):
        messages = {"messages": example["messages"]}
        formatted_chat = apply_chat_template(
            messages,
            tokenizer,
        )["text"]
        suffix_start = formatted_chat.rindex(example["suffix"].rstrip())
        prompt = formatted_chat[:suffix_start]
        completion = formatted_chat[suffix_start:]
        text = f"<MASK_START>{prompt}<MASK_END>{completion}"

        # Verify the tokens are present
        encoded = tokenizer(text)
        mask_start_id = tokenizer.convert_tokens_to_ids("<MASK_START>")
        mask_end_id = tokenizer.convert_tokens_to_ids("<MASK_END>")

        if mask_start_id not in encoded.input_ids or mask_end_id not in encoded.input_ids:
            raise ValueError("Mask tokens not found in encoded text!")

        return encoded

    processed_dataset = dataset.map(
        process_dataset, remove_columns=dataset["train"].column_names, num_proc=training_args.dataset_num_proc
    )

    # Filter examples longer than max_seq_length
    num_rows = processed_dataset["train"].num_rows
    processed_dataset = processed_dataset.filter(
        lambda x: len(x["input_ids"]) <= training_args.max_seq_length, num_proc=training_args.dataset_num_proc
    )
    print(
        f"Filtered {num_rows - processed_dataset['train'].num_rows} examples longer than {training_args.max_seq_length} tokens."
    )

    print(f"=== FORMATTED SAMPLE === \n{tokenizer.decode(processed_dataset['train'][0]['input_ids'])}")

    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=processed_dataset[script_args.dataset_train_split],
        eval_dataset=(
            processed_dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None
        ),
        data_collator=MaskedCompletionDataCollator(
            response_template="<MASK_END>", instruction_template="<MASK_START>", tokenizer=tokenizer, mlm=False
        ),
        peft_config=get_peft_config(model_config),
        processing_class=tokenizer,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
