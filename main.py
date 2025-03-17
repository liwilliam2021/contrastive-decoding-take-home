import transformers as tr
import torch
import torch.nn.functional as F

import time

amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)

amateur_model = tr.AutoModelForCausalLM.from_pretrained(amateur_path)
expert_model = tr.AutoModelForCausalLM.from_pretrained(expert_path)

# Oops my local machine is an M3 MAC
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

expert_model.to(device)
amateur_model.to(device)

expert_model.eval()
amateur_model.eval()

user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
	scores,
	results,
	kFactor = 4,
) {
	for (const result of results) {
		const { first, second, outcome } = result;
		const firstScore = scores[first] ?? 1000;
		const secondScore = scores[second] ?? 1000;

		const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
		const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
		let sa = 0.5;
		if (outcome === 1) {
			sa = 1;
		} else if (outcome === -1) {
			sa = 0;
		}
		scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
		scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
	}
	return scores;
}\n```"""

prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message},
    ],
    add_generation_prompt=True,
    tokenize=False,
)


def contrastive_generation(
    amateur_model,
    expert_model,
    prompt,
    max_tokens,
    tau=1,
    alpha=0.1,
    device=device,
    use_kv_cache=False,  # Disabled for instability reasons, see response.md
) -> str:

    # Tokenize the prompt.
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_ids = input_ids

    with torch.no_grad():
        kvc_expert = None
        kvc_amateur = None

        for _ in range(max_tokens):
            if use_kv_cache:
                # Only pass the last token if using KV cache
                model_input = generated_ids[:, -1].unsqueeze(-1)
            else:
                # Otherwise passs whole sequence
                model_input = generated_ids

            # Expert model
            outputs_expert = expert_model(
                model_input,
                past_key_values=kvc_expert if use_kv_cache else None,
                use_cache=use_kv_cache,
            )
            logits_expert = outputs_expert.logits[:, -1, :]
            probs_expert = F.softmax(logits_expert, dim=-1)
            if use_kv_cache:
                kvc_expert = outputs_expert.past_key_values

            # Amateur model
            outputs_amateur = amateur_model(
                model_input,
                past_key_values=kvc_amateur if use_kv_cache else None,
                use_cache=use_kv_cache,
            )
            logits_amateur = outputs_amateur.logits[:, -1, :] / tau
            probs_amateur = F.softmax(logits_amateur, dim=-1)
            if use_kv_cache:
                kvc_amateur = outputs_amateur.past_key_values

            # Compute V_head mask
            max_prob = torch.max(probs_expert, dim=-1, keepdim=True)[0]
            threshold = alpha * max_prob
            v_head = probs_expert >= threshold

            v_head_idx = torch.nonzero(v_head.squeeze(0)).flatten()

            # Compute contrastive score
            log_p_expert = torch.log(probs_expert.squeeze(0)[v_head_idx])
            log_p_amateur = torch.log(probs_amateur.squeeze(0)[v_head_idx])
            candidate_scores = log_p_expert - log_p_amateur

            # Greedy selection
            best_idx = torch.argmax(candidate_scores)
            next_token = v_head_idx[best_idx].unsqueeze(0).unsqueeze(0)
            generated_ids = torch.cat((generated_ids, next_token), dim=1)

            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
                print("EOS")
                break

    output_text = tokenizer.decode(generated_ids.squeeze()[input_ids.shape[1] :])
    return output_text


if __name__ == "__main__":
    """
    Generated Output:
      This function `updateEloScores` takes three arguments:
      - `scores`: a dictionary or similar mapping type that maps player identifiers (typically user IDs or usernames) to Elo scores (ratings). Initially, these scores should be initialized to some starting value, but in this implementation, all players start with an Elo score of 1000.
      - `results`: a list or iterable containing game results in the form of tuples or other structured data that includes:
      - `first`: the ID of the first player in the game
    Time: 131.33 seconds
    """
    start = time.time()
    output = contrastive_generation(
        amateur_model, expert_model, prompt, max_tokens=100, use_kv_cache=False
    )
    print(output)
    print(f"Time taken: {time.time() - start:.2f} seconds")
