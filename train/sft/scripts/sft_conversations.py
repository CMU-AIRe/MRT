from datasets import load_dataset
from transformers import AutoTokenizer

from src import get_checkpoint
from src.chat_templates import LLAMA_3_TEMPLATE
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


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

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
    if tokenizer.chat_template is None:
        tokenizer.chat_template = LLAMA_3_TEMPLATE
        tokenizer.eos_token = "<|eot_id|>"

    tokenizer.pad_token = "<|end_of_text|>"

    dataset = load_dataset(script_args.dataset_name)

    def process_dataset(example):
        messages = example["messages"]
        return apply_chat_template({"messages": messages}, tokenizer=tokenizer)

    processed_dataset = dataset.map(process_dataset, num_proc=training_args.dataset_num_proc)
    # Tokenize
    processed_dataset = processed_dataset.map(
        lambda x: tokenizer(x["text"], truncation=False),
        num_proc=training_args.dataset_num_proc,
        remove_columns=processed_dataset["train"].column_names,
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

    # response_template for Llama-3.1-8B-Instruct
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, tokenizer=tokenizer, mlm=False
    )

    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=processed_dataset[script_args.dataset_train_split],
        eval_dataset=(
            processed_dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None
        ),
        peft_config=get_peft_config(model_config),
        data_collator=data_collator,
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
