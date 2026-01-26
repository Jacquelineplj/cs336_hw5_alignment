from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
import torch
import torch.nn.functional as F
from einops import rearrange

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
    labels = rearrange(labels, "batch_size  sequence_length-> (batch_size sequence_length)")
    log_probs = F.log_softmax(logits, dim = -1)
    log_probs = rearrange(log_probs, "batch_size sequence_length vocab_size-> (batch_size sequence_length) vocab_size")
    log_probs = log_probs[torch.arange(log_probs.shape[0]), labels]
    log_probs = rearrange(log_probs, "(batch_size sequence_length)-> batch_size sequence_length ", batch_size = input_ids.shape[0])
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

