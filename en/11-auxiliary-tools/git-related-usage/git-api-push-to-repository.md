# Git Api Push to Repository

**Due to security concerns, GitHub stopped supporting account passwords for authentication of Git manipulation starting from August 13, 2021.**

1. **Can it use the API?**
   - The "API" here refers to **Personal Access Token (PAT)**. The answer is **yes**. This is exactly the alternative password method recommended by GitHub. When you use a HTTPS URL (for example `https://github.com/user/repo.git`) When cloning, pushing, or pulling code, enter the generated PAT when prompted for a password.
2. **How to set the API (PAT) as an environment variable?**
   - Setting the PAT as an environment variable is mainly used in scenarios where **automated scripts or programs call the GitHub API**, not for daily use. `git push/pull` Command. For daily command-line operations, it is recommended to use **Git Credential Manager**.

------



## Method 1: Recommended Approach - Using Personal Access Token (PAT) with Git Credential Manager



This is the most suitable method for daily `git` manipulation on your computer. You only need to generate a PAT, enter it once when verification is required, and then Git will remember it.



### Step 1: Generate Personal Access Token (PAT)



1. Log in to your GitHub account.
2. Click on the avatar in the upper right corner and select **Settings**.
3. In the left menu, scroll to the bottom and select **Developer settings**.
4. In the left menu, choose **Personal access tokens** -> **Tokens (classic)**.
5. Click **Generate new token**, then select **Generate new token (classic)**.
6. **Note**: Give your Token a recognizable name, such as "My Laptop Git".
7. **Expiration**: For security, it is recommended to set an expiration time, such as 30 days or 90 days.
8. **Select scopes**: This is the most important step. For operations like cloning, pulling, and pushing code repositories, **you must check the `repo` option**. It includes all the permissions related to the repository.
9. Click **Generate token** at the bottom of the page.
10. **Copy the generated Token immediately!** This Token will only appear once and will disappear after page refresh. Please save it in a safe place.



#### Step 2: Use in the command line



Now, when you execute commands that require verification such as `git push` or `git pull`:

- **Username (Username)**: Enter your GitHub username.
- **Password (Password)**: **Paste the Personal Access Token (PAT) you just copied.**

In most operating systems (Windows, macOS, Linux), Git automatically uses the Credential Manager to cache your credentials. Once entered successfully, you don't need to enter it again in a short period of time.

------



### Method 2: Setting environment variables for scripts or special purposes



If you are writing a script that needs to interact with the GitHub API (for example, a script that automatically creates a repository), setting the PAT as an environment variable is a good practice.

**Note:** This method is not recommended for daily `git` commands, as the credential manager is safer and more convenient.



#### Setting up on Linux / macOS



You can set the environment variables in `~/.bashrc`, `~/.zshrc`, or `~/.profile` files, so they will automatically take effect every time a new terminal is opened.

1. Open your shell configuration file, such as `~/.bashrc`:

   Bash

   ```
   nano ~/.bashrc
   ```

2. Add the following line at the end of the file, and replace `your_personal_access_token` with your own PAT:

   Bash

   ```
   export GITHUB_TOKEN="your_personal_access_token"
   ```

  **It is strongly recommended** to use the complete Token starting with `ghp_` or `ghs_`.

3. Save and close the file. Then execute the following command to make the configuration take effect immediately:

   Bash

   ```
   source ~/.bashrc
   ```

4. Verify if the setup is successful:

   Bash

   ```
   echo $GITHUB_TOKEN
   ```

If your token can be seen, it means the setup is successful.



#### Setting up on Windows



1. Search for "environment variables" in the search bar, and then select "Edit System Environment Variables".
2. In the pop-up "System Properties" window, click the "Environment Variables..." button.
3. In the "User Variables" or "System Variables" section (it is recommended to choose "User Variables"), click "New...".
4. **Variable name**: `GITHUB_TOKEN`
5. **Variable value**: Paste your Personal Access Token (PAT).
6. Click "OK" repeatedly to close all windows.
7. **Reopen** a new Command Prompt (CMD) or PowerShell window, and the new environment variables will take effect.
8. Verify if the setting is successful:
   - In CMD: `echo %GITHUB_TOKEN%`
   - In PowerShell: `echo $env:GITHUB_TOKEN`

------



### Summary



- **Daily use of `git` command**: Choose **Method 1**. Generate a PAT and paste it when Git prompts for a password. Git's credential manager will automatically remember it for you.
- **Writing scripts or programs to call the API**: Choose **Method 2**. Set the PAT as an environment variable, and then read this variable in the code to use it.
- **Another option: SSH**: In addition to HTTPS + PAT, you can also use the SSH protocol. This method requires you to generate an SSH key pair and add the public key to your GitHub account. After setting it up, you won't need to enter any username or password/Token. This is also a popular and secure method.