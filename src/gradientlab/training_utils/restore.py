from pathlib  import Path
import torch

def restore_weights(
    dir: Path,
    model: torch.nn.Module | None = None,
    optim=None,
    scaler=None,
    scheduler=None,
    submodules: list[str] | None = None,
):
    try:
        if model is not None:
            ckpt = torch.load(dir / "ckpt.pt", weights_only=True)
            if submodules is None:
                print("-- RESTORING WHOLE MODEL--")
                model.load_state_dict(ckpt, strict=False)
            else:
                print(f"-- RESTORING {submodules} ONLY--")
                model_state = model.state_dict()
                submodule_state = {
                    k: v
                    for k, v in ckpt.items()
                    if any(k.startswith(sm) for sm in submodules)
                }
                print(
                    f"restored: {len(list(submodule_state.keys()))}/{len(model_state)} modules"
                )
                model_state.update(submodule_state)
                model.load_state_dict(model_state, strict=False)
    except Exception as e:
        print(f"{e}")

    try:
        if optim is not None:
            optim.load_state_dict(torch.load(dir / "optim.pt", weights_only=True))
            print("restored optim state")
    except Exception as e:
        print(f"{e}")

    try:
        if scaler is not None:
            scaler.load_state_dict(
                torch.load(dir / "scaler.pt", weights_only=True),
            )
            print("restored scaler state")
    except Exception as e:
        print(f"{e}")
    try:
        if scheduler is not None:
            scheduler.load_state_dict(
                torch.load(dir / "scheduler.pt", weights_only=True),
            )
            print("restored scheduler state")
    except Exception as e:
        print(f"{e}")