"""All methods behind one interface: load_method(name, reso).edit(image, source_prompt, target_prompt, [mask])."""

METHODS = {  # name: (label, backbone, needs an edit mask, resolutions)
    "mln": ("MLN (ours)", "Switti", False, (512, 1024)),
    "aredit": ("AREdit", "Infinity-2B", False, (512, 1024)),
    "varin": ("VARIN (tau 18)", "HART-0.7B", False, (1024,)),
    "varin_tau9": ("VARIN (tau 9)", "HART-0.7B", False, (1024,)),
    "aredit_mask": ("AREdit (mask)", "Infinity-2B", True, (512, 1024)),
    "bitresedit": ("BitResEdit (mask)", "Infinity-2B", True, (512, 1024)),
    "bitresedit_lock": ("BitResEdit + structure lock (mask)", "Infinity-2B", True, (512, 1024)),
}


def load_method(name, reso):
    if reso not in METHODS[name][3]:
        raise ValueError(f"{name} runs at {METHODS[name][3]} px")
    if name == "mln":
        from methods.mln import MLN
        return MLN(reso)
    if name.startswith("aredit"):
        from methods.aredit import AREdit
        return AREdit(reso)
    if name.startswith("bitresedit"):
        from methods.bitresedit import BitResEdit
        return BitResEdit(reso, structure_lock=name == "bitresedit_lock")
    from methods.varin import VARIN
    return VARIN(tau=9.0 if name == "varin_tau9" else 18.0)
