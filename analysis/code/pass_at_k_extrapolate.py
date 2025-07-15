import argparse, pickle, os, json, sys
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from collections import defaultdict
from transformers import AutoTokenizer

CONTINUE_STRS = ["Wait", "But wait", "Alternatively", "But hold on"]


def remove_unfinished_line(text):
    lines = text.split("\n\n")
    lines_without_last = lines[:-1]
    return "\n\n".join(lines_without_last) + "\n"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_name", type=str, default='KbsdJames/Omni-MATH')
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--dataset_start", type=int, default=0)
    parser.add_argument("--dataset_end", type=int, default=1)
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("-K", type=int, default=4)
    parser.add_argument("-W", type=int, default=8)
    parser.add_argument("-R", type=int, default=8)
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)

    args = parser.parse_args()
    print(args.dataset_start, args.dataset_end, args.W)

    os.makedirs(args.output_path, exist_ok=True)

    dataset = load_dataset(args.input_dataset_name, split=args.dataset_split)
    args.dataset_end = min(args.dataset_end, len(dataset))

    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, enable_prefix_caching=True, max_model_len=40000)
    tokenizer = llm.get_tokenizer()

    prefixes = defaultdict(list)
    rollouts = defaultdict(list)

    # generate K thoughts for each problem
    for n in tqdm(range(args.dataset_start, args.dataset_end)):
        completions = llm.chat(
            messages=[
                {"role": "user", "content": dataset[n]['problem']}
            ],
            sampling_params=SamplingParams(
                n=args.K,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stop_token_ids=[151649],
                top_p=args.top_p,
            ),
            add_generation_prompt=True,
            continue_final_message=False,
        )
        assert len(completions[0].outputs) == args.K

        for output in completions[0].outputs:
            prefixes[n].append(output.text)
        
        if args.R > 0 and args.W > 0:
            # rollout without any waits
            completions = llm.chat(
                messages=[
                    [
                        {"role": "user", "content": dataset[n]['problem']},
                        {"role": "assistant", "content": prefixes[n][k]}
                    ] for k in range(args.K)
                ],
                sampling_params=SamplingParams(
                    n=args.R,
                    temperature=args.temperature,
                    max_tokens=4096, # each budget forcing step gets 2048 max tokens
                    top_p=args.top_p,
                ),
                add_generation_prompt=False,
                continue_final_message=True,
            )
            rollouts[n, 0] = completions

            # for each thought, add wait W times
            for w in range(args.W):
                completions = llm.chat(
                    messages=[
                        [
                            {"role": "user", "content": dataset[n]['problem']},
                            {"role": "assistant", "content": prefixes[n][k] + "\n" + CONTINUE_STRS[w % 4]}
                        ] for k in range(args.K)
                    ],
                    sampling_params=SamplingParams(
                        n=1,
                        temperature=args.temperature,
                        max_tokens=2048, # each budget forcing step gets 2048 max tokens
                        stop_token_ids=[151649],
                        top_p=args.top_p,
                    ),
                    add_generation_prompt=False,
                    continue_final_message=True,
                )

                for k, outputs in enumerate(completions):
                    output = outputs.outputs[0]
                    if output.finish_reason == 'length':
                        prefixes[n][k] += "\n" + CONTINUE_STRS[w % 4] + remove_unfinished_line(output.text)
                    elif output.finish_reason == 'stop':
                        prefixes[n][k] += "\n" + CONTINUE_STRS[w % 4] + output.text
                    else:
                        raise ValueError(f"Unexpected stop reason: {output.finish_reason}")

                # rollout every 2 extrapolations
                if (w + 1) % 2 == 0:
                    completions = llm.chat(
                        messages=[
                            [
                                {"role": "user", "content": dataset[n]['problem']},
                                {"role": "assistant", "content": prefixes[n][k]}
                            ] for k in range(args.K)
                        ],
                        sampling_params=SamplingParams(
                            n=args.R,
                            temperature=args.temperature,
                            max_tokens=4096,
                            top_p=args.top_p,
                        ),
                        add_generation_prompt=False,
                        continue_final_message=True,
                    )
                    rollouts[n, w + 1] = completions

    if args.R > 0 and args.W > 0:
        with open(os.path.join(args.output_path, f'extrapolate_{args.dataset_start}_{args.dataset_end}.pkl'), 'wb') as f:
            pickle.dump(rollouts, f)

    with open(os.path.join(args.output_path, f'pass_at_k_{args.dataset_start}_{args.dataset_end}.pkl'), 'wb') as f:
        pickle.dump(prefixes, f)

