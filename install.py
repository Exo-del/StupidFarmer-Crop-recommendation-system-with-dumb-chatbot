#!/usr/bin/env python3
"""Install required Python dependencies for CRS_V1."""

from __future__ import annotations
import subprocess
import sys

REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "joblib",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "xgboost",
    "llama-cpp-python",
]

STANDARD_LIBRARY_PACKAGES = [
    "tkinter",
    "warnings",
]


def run_pip_install(packages: list[str], extra_args: list[str] | None = None) -> int:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--no-cache-dir",
        "--prefer-binary",
    ]
    if extra_args:
        command.extend(extra_args)
    command.extend(packages)

    try:
        subprocess.check_call(command)
        return 0
    except subprocess.CalledProcessError as exc:
        print("\nERROR: Installation failed.")
        print("Command:", " ".join(command))
        print("Exit code:", exc.returncode)
        return exc.returncode


def main() -> int:
    print("Installing required packages for GUI and notebook...")
    print("Detected Python executable:", sys.executable)
    print("The following packages will be installed/updated:")
    for pkg in REQUIRED_PACKAGES:
        print(" -", pkg)
    print("\nNote: tkinter is provided by the Python standard library and is not installed via pip.")

    standard_packages = [pkg for pkg in REQUIRED_PACKAGES if pkg != "llama-cpp-python"]
    status = run_pip_install(standard_packages)
    if status != 0:
        return status

    print("\nInstalling llama-cpp-python separately with binary preference...")
    status = run_pip_install(["llama-cpp-python"])
    if status != 0:
        print("\n⚠️ llama-cpp-python installation failed. The GUI can still run without LLM explanations.")
        print("Try installing llama-cpp-python manually with:")
        print(f"  {sys.executable} -m pip install --upgrade --no-cache-dir --prefer-binary llama-cpp-python")
        print("If that fails, use a Python 3.11 or 3.12 environment for better wheel support.")
    else:
        try:
            import llama_cpp  # type: ignore
            print("✅ llama_cpp import succeeded.")
        except Exception as exc:
            print("\n⚠️ Installed llama-cpp-python, but llama_cpp could not be imported.")
            print("Error:", exc)
            print("Make sure the GUI uses the same Python interpreter used for installation.")

    print("\nInstallation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
