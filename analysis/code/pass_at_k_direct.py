import argparse, pickle, os
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams


def apply_template(problem):
    messages = [
        {"role": "user", "content": problem}
    ]
    return messages

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_name", type=str, default='qq8933/MATH500')
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--dataset_start", type=int, default=0)
    parser.add_argument("--dataset_end", type=int, default=500)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("-K", type=int, default=1)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)

    args = parser.parse_args()
    print(args.dataset_start, args.dataset_end)
    print(args.model)
    print(args.output_path)

    os.makedirs(args.output_path, exist_ok=True)

    dataset = load_dataset(args.input_dataset_name, split=args.dataset_split)
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, enable_prefix_caching=True, max_model_len=20000)
    args.dataset_end = min(args.dataset_end, len(dataset))

    solutions = {}
    for i in tqdm(range(args.dataset_start, args.dataset_end, args.batch_size)):
        batch_problems = dataset[i : min(i + args.batch_size, args.dataset_end)]['problem']
        
        convs = [apply_template(problem) for problem in batch_problems]
        completions = llm.chat(
            messages=convs,
            sampling_params=SamplingParams(
                n=args.K,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
            )
        )
        for j, completion in enumerate(completions):
            solutions[i + j] = completion

    with open(os.path.join(args.output_path, f'pass_at_k_direct_{args.dataset_start}_{args.dataset_end}.pkl'), 'wb') as f:
        pickle.dump(solutions, f)
