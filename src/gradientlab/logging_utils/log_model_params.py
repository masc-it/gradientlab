from torch import nn


def pretty_print_model(module: nn.Module):
    """
    Given a nn.Module, prints a formatted table listing each direct submodule's name and
    number of parameters (avoiding double-counting shared parameters), along with total
    trainable and non-trainable parameter counts.

    Args:
        module (nn.Module): A PyTorch nn.Module instance.
    """
    data = []
    seen_params = set()
    total_trainable = 0
    total_non_trainable = 0

    # ---------- 1.  parameters sitting *on the root module itself* ----------
    root_total = root_trainable = 0
    for p in module.parameters(recurse=False):
        pid = id(p)
        if pid not in seen_params:  # still guard against sharing
            seen_params.add(pid)
            n = p.numel()
            root_total += n
            if p.requires_grad:
                root_trainable += n
    if root_total:  # only add the row if there are any
        data.append(("root_name", root_total))
        total_trainable += root_trainable
        total_non_trainable += root_total - root_trainable

    # ---------- 2.  direct children (unchanged logic) -----------------------
    for name, submodule in module.named_children():
        sub_trainable = sub_total = 0
        for p in submodule.parameters():
            pid = id(p)
            if pid not in seen_params:
                seen_params.add(pid)
                n = p.numel()
                sub_total += n
                if p.requires_grad:
                    sub_trainable += n

        if sub_total:  # skip empty containers
            data.append((name, sub_total))
            total_trainable += sub_trainable
            total_non_trainable += sub_total - sub_trainable

    # Add total summary rows
    total_rows = [
        ("Total trainable params", total_trainable),
        ("Total non-trainable params", total_non_trainable),
    ]

    all_names = [name for name, _ in data] + [name for name, _ in total_rows]
    all_params = [num for _, num in data] + [num for _, num in total_rows]

    name_col_width = max(len("Module Name"), max(len(name) for name in all_names))
    param_col_width = max(
        len("Number of Parameters"), max(len(f"{n:,}") for n in all_params)
    )

    header = f"{'Module Name'.ljust(name_col_width)} | {'Number of Parameters'.rjust(param_col_width)}"
    separator = f"{'-' * name_col_width}-+-{'-' * param_col_width}"

    rows = [header, separator]
    for name, num in data:
        rows.append(f"{name.ljust(name_col_width)} | {num:>{param_col_width},}")

    rows.append(separator)
    for name, num in total_rows:
        rows.append(f"{name.ljust(name_col_width)} | {num:>{param_col_width},}")

    print("\n".join(rows))
    return total_trainable
