# Xget Library

There are mainly two ways to use Xget to download dataset and model from Hugging Face (HF):

## Method 1: Directly convert the URL for download (suitable for command-line tools)

This method is very suitable for when you need to directly download a specific model or dataset file. You just need to follow a simple rule: add the domain of Xget and the `/hf` prefix in front of the original Hugging Face URL.

**Conversion rules:**

* **Original URL**: `https://huggingface.co/...`
* **Accelerated URL**: `https://xget.xi-xu.me/hf/...`

**Example:**

1. **Download model file**
   * **Original URL**:
     ```
     https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/pytorch_model.bin
     ```
   * **URL after using Xget for acceleration**:
     ```
     https://xget.xi-xu.me/hf/microsoft/DialoGPT-medium/resolve/main/pytorch_model.bin
     ```
2. **Download dataset file**
   * **Original URL**:
     ```
     https://huggingface.co/datasets/rajpurkar/squad/resolve/main/plain_text/train-00000-of-00001.parquet
     ```
   * **URL after using Xget for acceleration**:
     ```
     https://xget.xi-xu.me/hf/datasets/rajpurkar/squad/resolve/main/plain_text/train-00000-of-00001.parquet
     ```

You can directly paste the converted URL into a browser, or use command-line download tools such as `wget`, `curl`, and `aria2` to download it. In the document, `aria2` is particularly recommended for multi-thread downloading to achieve the best speed:

**Bash**

```
# 使用 aria2 以 16 个线程下载大模型文件
aria2c -x 16 -s 16 https://xget.xi-xu.me/hf/microsoft/DialoGPT-large/resolve/main/pytorch_model.bin
```

### Method 2: Integrate into Python code (Recommended)

If you use libraries such as `transformers` or `datasets` in a Python project, the easiest way is to configure environment variables so that these libraries automatically download resources via Xget.

**Core steps:**

By setting the `HF_ENDPOINT` environment variable, all requests to the Hugging Face Hub are directed to the Xget image.

**Example Python code:**

The following code demonstrates how to set environment variables before loading the model. The `transformers` library will automatically use Xget for downloading.

**Python**

```
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# 关键步骤：设置环境变量，让 transformers 库自动使用 Xget 镜像
os.environ['HF_ENDPOINT'] = 'https://xget.xi-xu.me/hf'

# 像往常一样定义模型名称
model_name = 'microsoft/DialoGPT-medium'

print(f"正在从 Xget 镜像下载模型: {model_name}")

# 使用 from_pretrained 方法加载模型和分词器
# 由于设置了环境变量，代码无需任何其他修改
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("模型和分词器加载成功！")

# 接下来就可以正常使用模型了
# chat_history_ids = model.generate(...)
```

**Summary:**

* **Temporary download of a single file**: Manually convert the URL and use a download tool to obtain the file.
* **Long-term use in Python projects**: Set the `HF_ENDPOINT` environment variable at the beginning of the code, so the Hugging Face-related libraries can automatically benefit from acceleration without any changes to subsequent code.
