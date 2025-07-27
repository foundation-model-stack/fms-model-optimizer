# Standard
import os
import subprocess

dependencies_with_patch = {
    "microxcaling": "https://github.com/microsoft/microxcaling.git",
}


def install_with_patch(
    pkg_name: str,
    repo_url: str,
    patch_file: str,
    home_dir: str = None,
) -> None:
    """
    Install a dependency with a patch file

    Args:
        pkg_name (str): Name of package being installed
        repo_url (str): Github repo URL
        patch_file (str): Patch file in patches/<patch_file>
        home_dir (str): Home directory with fms-model-optimizer and other packages.
            Defaults to None.
    """
    # We want to git clone the repo to $HOME/repo_name
    if home_dir is None:
        home_dir = os.path.expanduser("~")

    # Get fms_mo directory in home_dir
    cwd = os.getcwd()

    # Get patch file location from fms-model-optimizer
    patch_file = os.path.join(cwd, "patches", patch_file)
    if not os.path.exists(patch_file):
        raise FileNotFoundError(f"Can't find {pkg_name} patch file in {cwd}/patches")

    # Check to see if package exists in cwd or home_dir
    pkg_path_cwd = os.path.join(cwd, pkg_name)
    pkg_path_home = os.path.join(home_dir, pkg_name)
    pkg_exists_cwd = os.path.exists(pkg_path_cwd)
    pkg_exists_home = os.path.exists(pkg_path_home)

    # If pkg already exists in cwd or home_dir, skip clone
    if pkg_exists_cwd:
        pkg_dir = pkg_path_cwd
        print(f"Directory {pkg_dir} already exists.  Skipping download.")
    elif pkg_exists_home:
        pkg_dir = pkg_path_home
        print(f"Directory {pkg_dir} already exists.  Skipping download.")
    else:
        # Clone repo to home directory
        pkg_dir = pkg_path_home
        subprocess.run(["git", "clone", repo_url], cwd=home_dir, check=True)

    # Apply patch and pip install package
    try:
        subprocess.run(["git", "apply", "--check", patch_file], cwd=pkg_dir, check=True)
        subprocess.run(["git", "apply", patch_file], cwd=pkg_dir, check=True)
        print(
            f"FMS Model Optimizer patch for {pkg_name} applied.  Installing package now."
        )
        subprocess.run(["pip", "install", "."], cwd=pkg_dir, check=True)

    except subprocess.CalledProcessError as e:
        print(
            f"FMS Model Optimizer patch for {pkg_name} is already installed "
            f"or an error has occured: \n{e}"
        )


def install_dependencies_with_patch() -> None:
    """
    Script to install depenencies that requires a patch prior to pip install.

    To execute, use `python install_patches.py`.

    Requirements:
        1. The patch file is named <package>.patch
        2. Patch file must be located in fms-model-optimizer/patches
    """
    for pkg, repo_url in dependencies_with_patch.items():
        install_with_patch(
            pkg_name=pkg,
            repo_url=repo_url,
            patch_file=pkg + ".patch",
        )


if __name__ == "__main__":
    install_dependencies_with_patch()
