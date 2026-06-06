import torch 
import torch.nn as nn
import torch.nn.functional as F

from models.helpers import sample_with_top_k_top_p_



class SampleSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits_BlV, rng, top_k, top_p, vae_quant_embed):
        ctx.save_for_backward(logits_BlV, vae_quant_embed.weight)
        logits_BlV = logits_BlV.detach()  # d   etach logits to avoid backprop through sampling
        idx_Bl = _actual_sample_logic(
            logits_BlV,
            rng,
            top_k,
            top_p,
            num_samples=1,  # we sample one index per position
                )
        h_BChw = vae_quant_embed(idx_Bl)  # get the quantized embeddings for the sampled indices
        ctx.mark_non_differentiable(idx_Bl)  # mark idx_Bl as non-d
        ctx.save_idx_Bl = idx_Bl  # save the sampled indices for backward pass
        return h_BChw  # return the quantized embeddings and indices
    @staticmethod
    def backward(ctx, grad_output_idx_Bl):
        logits_BlV, vae_quant_embedding_weight = ctx.saved_tensors
        idx_Bl = ctx.save_idx_Bl  # retrieve the sampled indices
        
        
        grad_h_BChw = grad_output_idx_Bl 

        grad_probs_soft = torch.matmul(grad_h_BChw, vae_quant_embedding_weight.T)
        probs = F.softmax(logits_BlV, dim=-1) 
        
        grad_logits = grad_probs_soft - probs * grad_probs_soft.sum(dim=-1, keepdim=True)

        return grad_logits, None, None, None, None 

def _actual_sample_logic(logits_detached: torch.Tensor, rng: torch.Generator, top_k: int = 0, top_p: float = 0.0, num_samples: int = 1) -> torch.Tensor:
    """
    Performs actual top-k/top-p filtering and multinomial sampling on DETACHED logits.
    This function is a direct adaptation of the user-provided `sample_with_top_k_top_p_`.
    It operates on detached logits; the STE wrapper handles gradient tracking.
    """
    B, l, V = logits_detached.shape
    
    # Apply deterministic re-seeding if desired (from your original function)
    # Be aware that this re-seeds for every call, making sampling deterministic
    # *for the same initial seed* across separate calls to this function.
    rng.manual_seed(rng.initial_seed()) 

    # Work on a clone to ensure original detached_logits_for_sampling is not altered
    # in case it's used elsewhere (though in this specific STE, it's just for sampling).
    processed_logits = logits_detached.clone()

    if top_k > 0:
        # Get the top_k values
        # [0] gets the values, [1] gets the indices (we only need values here)
        top_k_values = processed_logits.topk(top_k, largest=True, sorted=False, dim=-1)[0]
        
        # Get the minimum value among the top_k
        min_top_k_value = top_k_values.amin(dim=-1, keepdim=True)
        
        # Create a mask for elements to remove (those smaller than the k-th largest)
        idx_to_remove = processed_logits < min_top_k_value
        
        # Fill masked elements with -torch.inf
        processed_logits.masked_fill_(idx_to_remove, -torch.inf)
        
    if top_p > 0:
        # Sort logits in ascending order (your original code uses descending=False)
        sorted_logits, sorted_idx = processed_logits.sort(dim=-1, descending=False)
        
        # Compute softmax and cumulative sum on the sorted logits
        # .cumsum_() is in-place
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) 
        
        # Determine indices to remove: where cumulative probability is <= (1 - top_p)
        # This keeps the "nucleus" of top_p probability mass.
        sorted_idx_to_remove = cumulative_probs <= (1 - top_p)
        
        # Ensure the last element (largest probability in ascending sort) is never removed
        # (or more generally, ensure elements within the nucleus are not removed)
        sorted_idx_to_remove[..., -1:] = False
        
        # Scatter the `sorted_idx_to_remove` mask back to the original (unsorted) order
        # This creates a boolean mask `mask_to_fill` of the original logits shape
        mask_to_fill = sorted_idx_to_remove.scatter(
            sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove
        )
        
        # Fill masked elements with -torch.inf
        processed_logits.masked_fill_(mask_to_fill, -torch.inf)
        
    # Sample using multinomial
    # Replacement logic as per your original function
    replacement = num_samples >= 0 
    num_samples_abs = abs(num_samples) # Use abs for multinomial
    
    # Reshape for torch.multinomial (which expects 2D: (batch_size * sequence_length, vocab_size))
    # Then apply softmax on these processed logits
    probabilities_for_sampling = processed_logits.softmax(dim=-1).view(-1, V)
    
    # Perform multinomial sampling
    sampled_indices_flat = torch.multinomial(
        probabilities_for_sampling,
        num_samples=num_samples_abs,
        replacement=replacement,
        generator=rng,
    )
    
    # Reshape back to (B, l, num_samples)
    return sampled_indices_flat.view(B, l, num_samples_abs)
