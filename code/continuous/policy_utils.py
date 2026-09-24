import torch


def save_checkpoint(
    policy, vnet, occ, pi_opt, vf_opt, occ_opt, it, beta, cfg, save_dir
):
    """Save model checkpoint."""
    import os

    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "iteration": it,
        "beta": beta,
        "config": cfg.__dict__,
        "policy_state_dict": policy.state_dict(),
        "vnet_state_dict": vnet.state_dict(),
        "occ_state_dict": occ.state_dict(),
        "pi_opt_state_dict": pi_opt.state_dict(),
        "vf_opt_state_dict": vf_opt.state_dict(),
        "occ_opt_state_dict": occ_opt.state_dict(),
    }

    checkpoint_path = f"{save_dir}/checkpoint_iter_{it:05d}.pt"
    torch.save(checkpoint, checkpoint_path)

    # Also save as "latest"
    latest_path = f"{save_dir}/checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)

    print(f"  Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(
    checkpoint_path, policy, vnet, occ, pi_opt=None, vf_opt=None, occ_opt=None
):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)

    policy.load_state_dict(checkpoint["policy_state_dict"])
    vnet.load_state_dict(checkpoint["vnet_state_dict"])
    occ.load_state_dict(checkpoint["occ_state_dict"])

    if pi_opt is not None:
        pi_opt.load_state_dict(checkpoint["pi_opt_state_dict"])
    if vf_opt is not None:
        vf_opt.load_state_dict(checkpoint["vf_opt_state_dict"])
    if occ_opt is not None:
        occ_opt.load_state_dict(checkpoint["occ_opt_state_dict"])

    it = checkpoint["iteration"]
    beta = checkpoint["beta"]

    print(f"Loaded checkpoint from iteration {it}")
    return it, beta
