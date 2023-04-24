
import torch
from torch.distributions import Distribution

from copy import deepcopy



def test_all_models_minimal_functionality(
    model, batch_shape, sampling_batch, device
):
    # Tests arbitrary batch support  
    net = model.to(device)
    input_dim = model.input_dim
    output_dim = model.output_dim

    x = torch.randn(batch_shape + (input_dim,), device=device)
    p = net(x)

    assert isinstance(p, Distribution), "Your output must be a distribution..."

    # Test sampling with wierd batches...
    samples_single = p.sample()
    samples_batched = p.sample(sampling_batch)  # type: ignore

    assert (
        samples_single.shape == torch.Size(batch_shape) or samples_single.shape[-1] == output_dim or samples_single.shape.numel() == torch.Size(batch_shape + (output_dim,)).numel()
    ), f"Output dim does not match, using sample() it has shape {samples_single.shape}"
    assert (
        samples_batched.shape== torch.Size(sampling_batch + batch_shape) or
        (samples_batched.shape[-1] == output_dim and  samples_batched.shape[:-1]== torch.Size(sampling_batch + batch_shape))
        or samples_batched.shape.numel() == torch.Size(sampling_batch + batch_shape + (output_dim, )).numel()
    ), f"Output dim does not match, using sample(shape) it has shape {samples_batched.shape} expected {torch.Size(sampling_batch + batch_shape)}"

    # Test logprobs

    log_probs2 = p.log_prob(samples_batched)

    assert torch.isfinite(
        log_probs2
    ).all(), "Log probability of own samples should be finite"

    state_dict = net.state_dict()

    net = deepcopy(net)
    list(net.parameters())[0].data = torch.zeros_like(list(net.parameters())[0].data)

    net.load_state_dict(state_dict)

    p_new = net(x)
    log_probs3 = p_new.log_prob(samples_batched)

    assert torch.isclose(
        log_probs2, log_probs3, atol=1e-3
    ).all(), "Reloading with statedict does not work"


