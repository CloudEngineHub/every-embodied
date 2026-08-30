# Clear Pip Installation Cache

Interrupted package installations can leave downloaded wheels and archives in the pip cache. Inspect the cache before removing it:

```bash
python -m pip cache dir
python -m pip cache info
```

Remove only cached artifacts with pip's own command:

```bash
python -m pip cache purge
```

This does not uninstall packages from the active environment. If disk usage remains high, inspect the environment directory and temporary directory separately. Avoid deleting an entire shared cache while another installation is running, because concurrent processes may still be reading those files.
