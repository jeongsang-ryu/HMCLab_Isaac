"""HMCLab_Isaac install verification.

Run after finishing INSTALL.md steps. Prints a checklist and exits non-zero on failure.
"""

import sys


def check(label: str, fn):
    try:
        result = fn()
        print(f"[OK] {label}: {result}")
        return True
    except Exception as e:
        print(f"[FAIL] {label}: {type(e).__name__}: {e}")
        return False


def main() -> int:
    ok = True
    ok &= check("Python", lambda: ".".join(map(str, sys.version_info[:3])))

    def _check_sim():
        import isaacsim  # noqa: F401
        from isaacsim import SimulationApp  # noqa: F401

        return "importable"

    ok &= check("Isaac Sim import", _check_sim)

    def _check_lab():
        import isaaclab

        return getattr(isaaclab, "__version__", "unknown")

    ok &= check("Isaac Lab", _check_lab)

    def _check_torch_cuda():
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        return f"torch {torch.__version__}, device: {torch.cuda.get_device_name(0)}"

    ok &= check("Torch + CUDA", _check_torch_cuda)

    def _check_hmclab():
        import hmclab_isaac

        return hmclab_isaac.__version__

    ok &= check("hmclab_isaac", _check_hmclab)

    print()
    print("SUCCESS" if ok else "FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
