## Patching Third Party Dependencies
Some dependencies clash with the current FMS Model Optimizer environment and we need to apply a patch.
To do this, we have provided a script in `fms-model-optimizer` named `install_patches.py`.

To run this script:
```
python3 install_patches.py
```

The following optional packages require a patch:
* `microxcaling`: Uses outdated versions of PyTorch-related packages

## Making a Patch File
To make a git diff patch file, first make your desired changes to the repository.  Then run
```
git diff > <package>.patch
```
Packages may include files that differ by white spaces even if you didn't change them.
To address this, add `--ignore-all-spaces` to the `git diff` command.

To test the patch file, copy the `<package>.patch` file to `fms-model-optimizer/patches`.
Next add a new entry to the `install_patches.py` dictionary called `dependencies_with_patch` with the package name and repo URL:
```
dependencies_with_patch = {
    <package>: <URL>, # for <package>.patch
}
```
Lastly, run the python command for `install_patches.py`.