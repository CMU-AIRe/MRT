import argparse, pickle, os
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from collections import defaultdict

def apply_template(tokenizer, problem, prefix):

    conv = [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": prefix},
    ]

    text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False, continue_final_message=True)
    text += """
</think>"""
    return text[21:] # prevent double-bos

def apply_stop_template(tokenizer, problem, prefix):

    conv = [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": prefix},
    ]

    text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False, continue_final_message=True)
    text += """Time is up.

Given the time I've spent and the approaches I've tried, I should stop thinking and formulate a final answer based on what I already have.
</think>

**Step-by-Step Explanation and Answer:**

1. **"""
    return text[21:] # prevent double-bos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_name", type=str, default='Maxwell-Jia/AIME_2024')
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_start", type=int, default=0)
    parser.add_argument("--dataset_end", type=int, default=15)
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("-K", type=int, default=8)
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--prefix_file", type=str, default="")

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    dataset = load_dataset(args.input_dataset_name, split=args.dataset_split)
    with open(args.prefix_file, 'rb') as f:
        k_prefixes = pickle.load(f)
    args.dataset_end = min(args.dataset_end, len(dataset))
    print(args.dataset_start, args.dataset_end)


    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, enable_prefix_caching=True, max_model_len=45000, disable_custom_all_reduce=True)
    
    tokenizer = llm.get_tokenizer()

    solutions = defaultdict(list) # n -> [[prefix 0 -> [r1, r2, ...]], [prefix 0 -> [r1, r2, ...]], ...]
    for n in tqdm(range(args.dataset_start, args.dataset_end)):
        problem = dataset[n]['problem']
        
        for k in range(len(k_prefixes[n])):

            texts = [apply_stop_template(tokenizer, problem, "")]
            for i in range(len(k_prefixes[n][k]) - 1):
                texts.append(apply_stop_template(tokenizer, problem, "\n\n".join(k_prefixes[n][k][:i+1]) + "\n\n"))
            texts.append(apply_template(tokenizer, problem, "\n\n".join(k_prefixes[n][k])))

            # [prefix 0 -> [r1, r2, ...], prefix 1 -> [r1, r2, ...], ...]
            completions_k = llm.generate(
                texts,
                sampling_params=SamplingParams(
                    n=args.K,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    top_p=args.top_p,
                    # prompt_logprobs=0,
                    logprobs=0,
                ),
            )

            assert len(completions_k) == len(k_prefixes[n][k]) + 1

            solutions[n].append(completions_k)
            
    with open(os.path.join(args.output_path, f'pass_at_k_given_prefix_{args.dataset_start}_{args.dataset_end}.pkl'), 'wb') as f:
        pickle.dump(solutions, f)
