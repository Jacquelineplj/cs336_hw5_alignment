from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
import torch
import torch.nn.functional as F
from einops import rearrange
from typing import Callable, Literal

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the cross entropy loss and backprop its gradients for a microbatch.
    """
    metadata = {"response_tokens":response_mask.to(torch.float16).sum(),"mask_fraction":response_mask.to(torch.float16).mean()}
    loss = -masked_normalize(policy_log_probs, response_mask, dim = -1, normalize_constant = normalize_constant)
    loss = loss.mean()
    loss = loss/gradient_accumulation_steps
    loss.backward()
    return loss, metadata


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    mask_tensor = mask*tensor
    if dim is not None:
        res = mask_tensor.sum(dim = dim)
    else:
        res = mask_tensor.sum()
    return res / normalize_constant

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    logits = model(input_ids).logits
    nll = F.cross_entropy(
        logits.transpose(1, 2),   # (B, V, L)
        labels,                   # (B, L)
        reduction="none",
        ignore_index=-100,        # 如果你 labels 里没有 -100 也没关系
    )
    log_probs = -nll              # (B, L)
    if return_token_entropy:
        entropy = compute_entropy(logits)
        return {
            "log_probs": log_probs,
            "token_entropy": entropy,
        }
    return {
        "log_probs": log_probs,
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the logits (i.e., entropy of the final dimension).
    logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
    containing unnormalized logits.

    Returns:
    torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
    prediction.
    """
    lse = torch.logsumexp(logits, dim = -1,keepdim = True)
    p = torch.exp(logits - lse)
    expected_logit = torch.sum(p * logits, dim = -1, keepdim = True)
    entropy = expected_logit - lse
    return -entropy.squeeze(-1)

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    prompt_and_output_lens = []
    prompt_ids = []
    response_ids = []
    labels = []
    response_mask = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_id = tokenizer.encode(prompt)
        response_id = tokenizer.encode(output)
        prompt_ids.append(torch.tensor(prompt_id))
        response_ids.append(torch.tensor(response_id))
        prompt_and_output_lens.append(len(prompt_id) + len(response_id))
    
    max_prompt_and_output_len = max(prompt_and_output_lens)
    input_ids = []
    labels = []
    response_mask = []

    for i, (prompt_id, response_id) in enumerate(zip(prompt_ids, response_ids)):
        input_id = torch.cat([prompt_id, response_id], dim = 0)
        mask = torch.cat([torch.zeros(len(prompt_id), dtype = torch.bool), torch.ones(len(response_id), dtype = torch.bool)], dim = 0)
        pad_len = max_prompt_and_output_len - len(input_id)
        input_ids_pad = F.pad(input_id, (0, pad_len), value = tokenizer.pad_token_id)
        mask_pad = F.pad(mask, (0, pad_len), value = False)
        input_ids.append(input_ids_pad[:-1])
        labels.append(input_ids_pad[1:])
        response_mask.append(mask_pad[1:])

    input_ids = torch.stack(input_ids, dim = 0)
    labels = torch.stack(labels, dim = 0)
    response_mask = torch.stack(response_mask, dim = 0)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }

def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, 
    normalized by the group size.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], 
            scores the rollout responses against the ground truths, 
            producing a dict with keys 
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. 
            The length of this list is 
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples. 
            The length of this list is `rollout_batch_size`, 
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,): 
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,): 
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    rewards = []
    metadata = {}
    for rollout_response, repeated_ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(rollout_response, repeated_ground_truth)
        rewards.append(reward_dict["reward"])
    rewards = torch.tensor(rewards, device = 'cuda')
    rewards = rearrange(rewards, "(n_prompts_per_rollout_batch group_size)-> n_prompts_per_rollout_batch group_size", group_size = group_size)
    mean_reward = rewards.mean(dim = -1, keepdim = True)
    std_reward = rewards.std(dim = -1, keepdim = True)
    normalized_rewards = rewards - mean_reward
    if normalize_by_std:
        normalized_rewards = normalized_rewards / (std_reward + advantage_eps)
    rewards = rearrange(rewards, "n_prompts_per_rollout_batch group_size-> (n_prompts_per_rollout_batch group_size)", group_size = group_size)
    normalized_rewards = rearrange(normalized_rewards, "n_prompts_per_rollout_batch group_size-> (n_prompts_per_rollout_batch group_size)", group_size = group_size)
    metadata = {
        "mean_reward": mean_reward.mean().item(),
        "std_reward": std_reward.mean().item()
    }
    return normalized_rewards, rewards, metadata



def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    metadata = {}
    r = torch.exp(policy_log_probs - old_log_probs)
    lhs = r * advantages
    rhs = torch.clamp(r, 1 - cliprange, 1 + cliprange) * advantages
    loss = -torch.min(lhs, rhs)
    metadata = {'is_clipped': lhs >= rhs}
    return loss, metadata



def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    if loss_type == "no_baseline":
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    elif loss_type == "grpo_clip":
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    mask_tensor = mask*tensor
    if dim is not None:
        res = mask_tensor.sum(dim = dim)/mask.sum(dim = dim)
    else:
        res = mask_tensor.sum()/mask.sum()
    return res


    
def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    loss, metadata = run_compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    loss = masked_mean(loss, response_mask, dim = -1)
    loss = loss.mean()
    loss = loss/gradient_accumulation_steps
    loss.backward()
    return loss, metadata


   
def grpo_microbatch_train_step_normalize(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    loss, metadata = run_compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    loss = masked_normalize(loss, response_mask, dim = -1)
    loss = loss.mean()
    loss = loss/gradient_accumulation_steps
    loss.backward()
    return loss, metadata