# Conda Env Specifies the Installation Path

## Method 1: Configure `envs_dirs` (Recommended)

Conda can store environments on a large data disk while still allowing activation by environment name. Define the storage directory once, then add it to `envs_dirs`.

**Operation steps:**

1. **Add the path to Conda configuration**:
    ```bash
    export ENV_STORAGE_ROOT="${ENV_STORAGE_ROOT:-$HOME/conda-envs}"
    mkdir -p "$ENV_STORAGE_ROOT"
    conda config --add envs_dirs "$ENV_STORAGE_ROOT"
    ```

2.  **Verify configuration**:
    ```bash
    conda info
    # Confirm that $ENV_STORAGE_ROOT is listed under "envs directories".
    ```

3. **Use the name to create an environment as usual**:
   Since the data disk path is at the first position, Conda will automatically create the new environment there by default.
    ```bash
    conda create -n openpi-server python=3.11 -y
    ```

4. **Activate with the name as usual**:
    ```bash
    conda activate openpi-server
    ```

In this way, both the storage space problem (actually on the data disk) is solved, and the concise command experience is maintained.

---

### Method 2: Setting Shell Alias (Alias)

If you do not want to modify the Conda configuration, add a shell function to `~/.bashrc`:

1.  Edit `.bashrc`:
    ```bash
    echo 'act_openpi() { conda activate "${ENV_STORAGE_ROOT:-$HOME/conda-envs}/openpi-server"; }' >> ~/.bashrc
    source ~/.bashrc
    ```
2. In the future, simply enter `act_openpi` to activate it.

**Summary**: Method 1 is recommended. This is a once-and-for-all solution, and other large environments created later can also be automatically saved to the data disk.
