# AUP Learning Cloud 科研文献检索实践记录：AMD ROCm、Qwen3-Embedding 与真实排错
环境搭建、依赖安装、模型部署、故障排查与结果复现
实践日期：2026年10月24日。

我在环境科学与工程方向做研究，经常需要从颗粒污泥文献中核对盐度、运行周期、碳源和反应器条件。这次课程实践，我选择了一个范围较小的任务：用预训练模型寻找论文原文，并与关键词检索进行对照。

下载的结果记录显示，本次课程运行使用10篇英文论文、103个PDF物理页和669段原文，通过 `Qwen/Qwen3-Embedding-0.6B` 生成了1024维索引及5道开发题、15道测试题的评测文件。669段文本的向量编码记录为 **97.26秒**。这些是本次导出记录中的结果，不是发布前重新测得的性能。

过程并不顺利：1.21 GB模型包尚未上传完成，服务器时长到期；改成云端下载后，又遇到了 `tokenizers` 导入失败和 `model_manifest.json` 缺失。下面记录实际经过，也提供整理后的操作命令；未在AUP重新执行的新增辅助工具会单独标明。

最终结果同样值得保留：15道测试题中，BM25和语义检索都找到了10题的目标页；语义检索的目标页排序更靠前，但严格原句命中数更低。这次实践的收获是得到一个能运行、能检查出处、能留下实验记录的小工具，以及对它适用范围的具体认识。

## 1. 这个项目具体解决什么问题

输入是英文研究问题，例如 `What feed salinity was maintained during reactivation after refrigeration?`，意思是“冷藏后复苏阶段采用了什么进水盐度”；输出是候选论文、PDF物理页码和原文片段，供研究者回到原文核对。

本次20题的检索输入均为英文。当前BM25脚本只按英文词项和数字分词，不适合直接用纯中文问题做公平对照；中文可作为自己的问题说明，检索输入需准备对应英文。

```text
论文PDF → 按页提取文字 → 切成原文片段
                              ├─ BM25关键词检索
                              └─ Qwen向量编码 → 语义检索
同一问题 → 各取前5段 → 检查目标页、原句与实验条件
```

BM25按照词项及其统计权重排序。语义检索把问题和段落编码成向量，再按照相似度排序。本项目没有训练或微调模型，也没有接入生成式回答模块。科研应用落在“帮助定位证据”这个环节，具体解释仍由研究者完成。

模型来源和用法可查阅[Qwen官方模型卡的本次固定版本](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/README.md)。

## 2. 实际使用的环境与参数

下面是下载结果包中记录的环境，适用于本次实践，不能据此保证所有账号都会分到相同设备。

| 项目 | 本次记录 |
|---|---|
| 平台与课程环境 | AUP Learning Cloud；Deep Learning Course |
| GPU设备名 | Radeon 8060S Graphics |
| PyTorch报告的GPU可见内存 | 64 GiB，接口报告值 |
| Python | 3.12.3 |
| PyTorch | 2.9.1+rocm7.13.0 |
| HIP版本字符串 | 7.13.99004-3309c6114a |
| Sentence-Transformers | 5.7.0 |
| Transformers / Tokenizers | 4.57.6 / 0.22.2 |
| Hugging Face Hub | 0.36.2 |
| JupyterLab / IPykernel | 4.5.6 / 6.31.0 |
| 模型 | Qwen/Qwen3-Embedding-0.6B |
| 模型提交 | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| 片段目标长度 / 重叠 | 1000字符 / 150字符，不跨PDF页 |
| 最大输入长度 / 批量大小 | 576 token / 4 |
| 检索数量 | Top-5 |

注意区分课程卡片中的系统内存与程序报告的GPU可见内存；这里记录的是 `torch.cuda.get_device_properties(0).total_memory` 的返回值，不能直接等同于独立显卡的物理显存配置。

576 token也不是看过检索成绩后选择的参数。本地检查发现最长片段为523 token，因而在观察检索排名前，把最初计划的512上限调整为576，保留全部669个片段。原文与带指令的问题均在索引前检查长度，避免静默截断。

## 3. 先弄清文件在哪里、代码在哪里运行

这次操作涉及三处界面：

| 界面 | 用途 |
|---|---|
| 本地Windows文件夹 | 保存准备材料，接收下载结果 |
| AUP网页中的左侧文件栏 | 上传、打开、下载云端文件 |
| AUP网页中的Terminal与Notebook | Terminal执行Linux命令；Notebook执行分格Python代码 |

本文标为 `bash` 的命令在 **AUP网页的Terminal** 中执行。标为 `python` 的代码放在 **Notebook代码格** 中执行。Notebook里点击代码框后按 `Shift+Enter`，等待运行结束再继续。

分享目录中的主要文件如下：

```text
README.md                         本文
evidence_search.py                原始检索脚本的副本
requirements-base.txt
requirements-cloud.txt
run_config.json                   本次固定参数
resources/model_manifest.json     原始模型包中的项目专用清单
scripts/install_model_manifest.py 整理教程时补充的清单校验工具
scripts/validate_inputs.py        自备文献与题集的输入核验工具
scripts/summarize_results.py      从私有结果包重算并导出数值摘要
results/run_summary.json          本次真实运行的数值摘要
results/per_question_metrics.csv  逐题定位指标，不含论文片段
results/observed_versions.txt     环境记录，不能当作通用安装命令
docs/运行记录核对说明.md           结果依据与留档范围
```

**数据与复现范围：** 我的课程私有启动包约60 MB，已经包含10篇PDF、原文片段、问题、核验脚本和 `01_AUP_GPU_run.ipynb`。本分享目录只提供程序、说明和数值记录；论文全文、提取原文、私有评测ZIP和模型权重不随分享包分发。已有课程启动包可直接沿用；其他读者可以使用自己有权处理的PDF，按第5节准备输入。更换文献和题集后，片段数量及成绩会变化。

公开目录不包含重算原10篇成绩所需的原始语料、参考标注与排名原文，因此其他读者不能仅凭这里的数值摘要独立验证原成绩。`resources/model_manifest.json`只列出公共模型文件名、大小和校验哈希，不包含权重或访问令牌；文中的`/home/jovyan`、`/opt/workspace/DL`和`$env:USERPROFILE`分别是通用课程目录或路径模板，没有写入个人用户名和私人路径。

发布范围应限于本分享目录。`.gitignore`只能帮助Git忽略尚未跟踪的文件，不能清理已经提交的内容或历史，也不会替网页手动上传和ZIP打包过滤文件。本次核查没有覆盖远程仓库与Git历史；重新加入数据、截图或日志后需要再次检查。

## 4. 启动AUP：先确认环境，再处理文件

### 4.1 登录与选择环境

使用老师或管理员提供的节点入口和账号登录。我使用的是课程账号；AUP本地账号的“用户名＋密码”入口与AMD官网的邮箱账号入口需要区分。是否可以使用GitHub登录，要看课程授权。

本次实际使用的GPU环境是 **Teaching Labs → Deep Learning Course**。环境名称可能随部署变化，应选择包含ROCm、PyTorch和JupyterLab的课程GPU环境。可选资源、运行时长和配额以账号页面为准；不能单凭环境名称判断哪个更便宜。[AUP官方操作指南](https://amdresearch.github.io/aup-learning-cloud/user-guide/platform-basics.html)

进入后可先在Terminal执行：

```bash
python3 - <<'PY'
import torch
print('PyTorch:', torch.__version__)
print('HIP:', torch.version.hip)
print('GPU available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
PY
```

这一步先确认当前环境能够识别AMD GPU；后面的 `evidence_search.py check` 还会实际进行小型矩阵计算。

### 4.2 第一个问题：`pwd`为什么显示 `/opt/workspace/DL`

我刚进入课程环境时，终端输出的是：

```text
/opt/workspace/DL
```

一开始我以为没有教程中的 `/home/jovyan` 目录。实际上，`pwd`只回答“现在在哪个目录”，不会列出其他目录。

在Terminal分别执行：

```bash
ls -ld /home/jovyan
cd /home/jovyan
pwd
```

最后应显示 `/home/jovyan`。AUP官方将其列为持久化用户目录；我把已上传完成的小代码包移到了这里。课程默认目录是否也持久化，需要另行确认。[文件保存说明](https://amdresearch.github.io/aup-learning-cloud/user-guide/platform-basics.html#save-important-files)

还有一个容易忽略的细节：**终端执行 `cd` 不会改变左侧文件栏的目录。** 上传文件前还要在左侧文件栏进入对应用户目录。文件栏显示的 `/` 也可能只是JupyterLab自己的根目录，不能仅凭这个符号判断Linux绝对路径。

### 4.3 第二个问题：1.21 GB模型包还没上传，服务器就到期了

原计划上传两个ZIP：60 MB代码与数据包，以及1.21 GB离线模型包。小包上传成功后，大包还没传完，服务器运行时长就结束了。

这时首先检查已保存在用户目录里的文件：

```bash
ls -lh /home/jovyan
```

服务器可以在有剩余配额时重新申请启动，但重新启动并不意味着浏览器会接着上传未完成的大文件。因此，我改用 **Plan B：让云服务器从官方仓库下载固定版本模型**。

下载和解压本身不需要GPU。之后复现时，可以先在CPU环境准备文件，再进入GPU环境计算；但这种做法有一个前提：确认两种环境共享同一持久化用户存储。对话里提出了这个建议，现有运行记录没有单独证明下载阶段最终切换到了哪一种CPU环境。

## 5. 准备代码与论文输入

### 5.1 已有课程私有启动包

在左侧文件栏上传小包后，在Terminal执行：

```bash
cd /home/jovyan
python3 - <<'PY'
from pathlib import Path
from zipfile import ZipFile
p = Path('AUP_代码与数据启动包.zip')
assert p.is_file(), '没有找到代码与数据启动包'
with ZipFile(p) as archive:
    archive.extractall('/home/jovyan')
assert Path('aup_evidence_project/01_AUP_GPU_run.ipynb').is_file()
print('代码与数据解压成功')
PY
```

项目应位于 `/home/jovyan/aup_evidence_project`。本文的 `scripts/` 和 `resources/` 是分享目录中的小文件夹，使用下文新增校验工具时，也把它们放到该项目中。

### 5.2 从本分享目录开始，使用自己的论文

把本目录中的代码放到云端的 `aup_evidence_project` 文件夹，在其中建立 `pdfs` 文件夹，放入有权使用的PDF。先在Terminal进入项目目录，安装文字提取依赖，再生成原文片段：

```bash
cd /home/jovyan/aup_evidence_project
python3 -m pip install -r requirements-base.txt
python3 evidence_search.py prepare --pdf-dir pdfs --chunk-size 1000 --overlap 150
```

它会生成 `private_work/pages.jsonl`、`chunks.jsonl` 和 `catalog.json`。页码从1开始，按PDF物理页计数，可能不同于期刊印刷页码。扫描PDF的OCR不包含在这个起步脚本中；提取为空或很短的页面需要单独检查。

评测题保存为 `questions.jsonl`，每行一个JSON对象。下面仅解释字段，示例ID和文字需要替换为自己从原文确认的内容：

```json
{"id":"T01","split":"test","query":"Your evidence-location question","gold_pages":["p_REPLACE_WITH_DOCUMENT_ID:p2"],"key_quote":"Replace with an exact sentence from that PDF page."}
```

`gold_pages`中的文档ID来自实际 `chunks.jsonl` 的 `doc_id`，不是论文标题，也不是本地文件序号。可以在Terminal查看已提取文献的ID：

```bash
python3 - <<'PY'
import json
from pathlib import Path
catalog = json.loads(Path('private_work/catalog.json').read_text(encoding='utf-8'))
for document in catalog['documents']:
    print(document['file'], document['doc_id'], 'PDF pages:', document['pdf_pages'])
PY
```

`split`使用 `dev` 或 `test`；本文的Notebook评测循环会运行两组，因此两组都要有问题。先核对原句和页码、固定题集，再观察检索排名。只想先试单题检索时，可先不建评测题集，按第8节的 `search` 示例运行。

原课程包中的这一步已经完成，共20题，分为5道开发题和15道测试题。自行准备的数据应使用自己的数量和出处，不套用本文的成绩。

## 6. Plan B：在云端下载固定版本模型

### 6.1 先测试访问，再下载

在Terminal执行：

```bash
python3 - <<'PY'
from urllib.request import urlopen
url = 'https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/config.json'
with urlopen(url, timeout=15) as response:
    print('HTTP:', response.status)
PY
```

HTTP 200表示配置文件可访问；大型权重文件仍需实际下载验证。

检查下载工具：

```bash
python3 -c "from huggingface_hub import snapshot_download; print('download tool ready')"
```

若工具缺失，可安装本次使用的版本：

```bash
python3 -m pip install "huggingface_hub==0.36.2"
```

然后执行：

```bash
cd /home/jovyan/aup_evidence_project
python3 - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

model_dir = Path('models/Qwen3-Embedding-0.6B')
snapshot_download(
    repo_id='Qwen/Qwen3-Embedding-0.6B',
    revision='97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3',
    local_dir=str(model_dir),
    max_workers=2,
)
for name in ['model.safetensors', 'tokenizer.json', 'config.json']:
    assert (model_dir / name).is_file(), name
print('官方模型文件下载完成：', model_dir.resolve())
PY
```

`revision`固定完整提交号；`local_dir`指定项目模型目录。Hugging Face下载工具会在本地目录中保存下载元数据，帮助识别已经下载的文件。若下载被中断，可以重新执行相同命令；单个大文件是否能够从原位置继续，仍取决于下载后端和缓存状态。[Hugging Face下载说明](https://huggingface.co/docs/huggingface_hub/v0.36.0/guides/download)

### 6.2 第三个问题：官方模型下载好了，为什么还缺 `model_manifest.json`

在后续最小模型推理时，我遇到了这个真实错误：

```text
FileNotFoundError: [Errno 2] No such file or directory:
'/home/jovyan/aup_evidence_project/models/Qwen3-Embedding-0.6B/model_manifest.json'
```

Notebook最后还显示了：

```text
CalledProcessError: ... 'evidence_search.py', 'smoke', ... returned non-zero exit status 1
```

应该向上查看子进程的具体异常。`CalledProcessError`只说明子程序失败，真正原因是前面的文件缺失。

检查脚本后可以看到：本项目一旦发现本地模型目录存在，就要求先读取 `model_manifest.json`，核对模型名称、提交版本、文件列表和大小。这个清单是原离线包附带的**项目专用文件**，不属于官方模型仓库必备文件。改变模型获取方式后，路径相同，项目的附加文件约定却没有自动满足。

对话中给出的修复方法，是从本地原模型ZIP里单独提取这个JSON，上传到模型目录，再运行 `smoke`。最终结果包确实记录了模型推理成功；具体上传操作没有单独留下终端日志，因此本文不把它扩写成一条已被完整记录的命令历史。

为方便后来者，分享目录附上了原清单，并新增一个校验工具。在项目目录执行：

```bash
python3 scripts/install_model_manifest.py --model-dir models/Qwen3-Embedding-0.6B
```

这个工具会逐个核对11个文件的大小和SHA256，全部一致后再把参考清单放进模型目录。它在整理教程时新增，已用本地固定版本模型验证，尚未重新在AUP执行；它不改变原始检索脚本和本次成绩。

原脚本加载时只检查清单中的身份、版本和文件大小；新增工具的完整SHA256校验比原加载检查更严格。清单里的 `inference_executed: false` 是10月23日准备阶段的记录字段，不表示10月24日云端推理未完成。运行是否成功应看 `model_smoke.json` 和索引记录。

## 7. 运行Notebook，并处理Python内核问题

### 7.1 初始化和GPU实际计算检查

已有课程私有启动包时，打开项目中的 `01_AUP_GPU_run.ipynb`，选择课程的Python内核，先运行初始化代码格。

**只有本分享代码的读者：** 左侧文件栏先进入 `aup_evidence_project`，点击上方“＋”打开Launcher，选择 **Python 3** 新建Notebook，将文件重命名为 `01_AUP_GPU_run.ipynb`。把第7、8节的Python代码按顺序分别粘贴到代码格中运行，最后按第10节另存结果。分享包没有附原课程Notebook，也没有预先生成你的文献数据。

初始化代码如下；原课程Notebook已经包含等价逻辑：

```python
from pathlib import Path
import json, subprocess, sys

import os
os.chdir('/home/jovyan/aup_evidence_project')
ROOT = Path.cwd()
assert (ROOT / 'evidence_search.py').is_file()
config = json.loads((ROOT / 'run_config.json').read_text(encoding='utf-8'))

def run(*args):
    return subprocess.run(
        [sys.executable, 'evidence_search.py', *args], cwd=ROOT, check=True
    )

print('Project:', ROOT)
print('Python:', sys.executable)
```

项目位置应为 `/home/jovyan/aup_evidence_project`。然后执行：

```python
run('check')
```

成功输出：

```text
AMD GPU check passed
CompletedProcess(..., returncode=0)
```

程序检查HIP版本、GPU可用性，并在GPU上计算一个16×16矩阵。它不是只读取显卡名称；通过记录会写入 `private_work/hardware.json`。

这里使用 `torch.cuda` 是正常的：PyTorch的ROCm后端沿用这套接口名称。AMD环境中看到 `device='cuda'`，不能据此判断调用了NVIDIA显卡。[PyTorch 2.9的HIP说明](https://docs.pytorch.org/docs/2.9/notes/hip.html)

### 7.2 安装附加依赖，保留课程PyTorch

在Notebook运行现成的安装格；等价代码为：

```python
import importlib.metadata

constraint = ROOT / 'private_work/torch-constraint.txt'
constraint.parent.mkdir(exist_ok=True)
constraint.write_text(
    'torch==' + importlib.metadata.version('torch') + '\n', encoding='utf-8'
)
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-r', 'requirements-cloud.txt',
    '--constraint', str(constraint), '--upgrade-strategy', 'only-if-needed'
], check=True)
run('check')
```

课程镜像已经提供ROCm版PyTorch，版本约束用于避免依赖安装时意外替换它。安装后再检查GPU。`results/observed_versions.txt`是本次软件版本的记录，里面的GPU构建版本不应直接拿到其他机器上盲目安装。

安装日志中曾出现两种容易被误解的提示：

```text
Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied
```

前者表示系统软件目录不可写，pip改用个人用户目录；后者表示对应依赖已经存在。是否成功，还要检查命令退出状态，以及当前内核能否实际导入这些包。

### 7.3 第四个问题：安装日志有Tokenizers，Notebook却找不到

实际报错出现在输入核验格：

```text
ModuleNotFoundError: No module named 'tokenizers'
```

但前面的安装日志已经显示 `tokenizers 0.22.2`。这说明“某次安装命令找到了包”与“当前内核能导入包”需要分别验证，不能仅凭安装日志判断。

对话中的处理顺序是：

1. 点击 **Kernel → Restart Kernel**，只重启Notebook内核。
2. 重新运行最上面的初始化格，恢复 `ROOT`、`config` 和 `run`。
3. 在代码格测试：

```python
import tokenizers
print('Tokenizers version:', tokenizers.__version__)
```

若仍然缺失，再在Notebook代码格安装：

```python
%pip install --no-deps "tokenizers==0.22.2"
```

随后再次重启内核、初始化并测试导入。`%pip`用于当前内核环境；它是IPython命令，只放在Notebook代码格中使用。[IPython说明](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-pip)

还未解决时，应记录解释器与模块路径，再继续定位：

```python
import sys, site, importlib.util
print('Python:', sys.executable)
print('User site:', site.getusersitepackages())
print('User site in path:', site.getusersitepackages() in sys.path)
print('Tokenizers:', importlib.util.find_spec('tokenizers'))
```

下载的Notebook保留了 `Tokenizers version: 0.22.2` 和随后输入核验通过的输出。现有记录没有逐条保留修复指令，因此可以确认最终恢复正常，但不能断言重启与重新安装两种办法都实际执行过，也不能把具体根因定为某一种内核路径问题。

### 7.4 确认输入，再做最小模型推理

原课程启动包的核验格最终打印：

```text
{'pdfs': 10, 'physical_pages': 103, 'chunks': 669,
 'questions': 20, 'source_quote_matches': 20}
```

这表示文件与参考原句匹配检查通过。参考标注由助手从源文核对，独立人工确认仍未完成。

使用自己文献与题集的读者，在安装依赖并完成第5节输入准备后，先在Terminal执行下面的新增核验命令：

```bash
cd /home/jovyan/aup_evidence_project
python3 scripts/validate_inputs.py --work private_work --questions questions.jsonl --tokenizer models/Qwen3-Embedding-0.6B/tokenizer.json --max-tokens 576
```

它检查问题ID、开发/测试分组、目标页、参考原句是否匹配提取页面、原句是否落在单个片段内，以及片段与带指令问题的token长度。通过后保存 `private_work/input_validation.json`。这些是输入一致性检查，不替代阅读PDF原页；若只做单题检索，可跳过题集核验和评测循环。

随后执行：

```python
run('smoke', '--revision', config['model_revision'])
```

真实成功输出为：

```text
AMD model smoke check passed: (2, 1024)
```

它编码两条自写短文本，用来确认模型能在当前GPU上运行。两个1024维向量不是准确率，也不是训练成果。

## 8. 建立669段原文索引，运行同一题集对照

在Notebook执行：

```python
run('index', '--revision', config['model_revision'],
    '--batch-size', str(config['batch_size']),
    '--max-tokens', str(config['max_input_tokens']))
```

本次索引记录：

| 项目 | 真实记录 |
|---|---:|
| 片段数量 | 669 |
| 向量维度 | 1024 |
| 批量大小 | 4 |
| 超过输入上限的片段 | 0 |
| 模型加载记录 | 0.40秒 |
| 文本向量编码记录 | 97.26秒 |

97.26秒对应代码中同步GPU前后计时的 `model.encode()` 阶段，不含下载、PDF提取、模型加载和后续评测。本次没有做CPU向量编码基准或多次重复计时，因此不报告GPU加速倍数。

复现者可以先在Terminal用同一英文问题查看两种检索结果：

```bash
python3 evidence_search.py search "What feed salinity was maintained during reactivation after refrigeration?" --method bm25 --top-k 5
python3 evidence_search.py search "What feed salinity was maintained during reactivation after refrigeration?" --method dense --top-k 5
```

输出中的 `file` 是论文文件名，`page` 是PDF物理页，`text` 是候选原文，`rank` 是排名。`score`用于方法内部排序，两种方法的分数不能直接比较，也不是答案正确的概率。

需要评测时，再回到Notebook运行：

```python
for method in ['bm25', 'dense']:
    for split in ['dev', 'test']:
        run('evaluate', '--questions', 'questions.jsonl',
            '--method', method, '--split', split,
            '--top-k', str(config['top_k']))
```

本次导出文件包含四组评测，每组都有JSON结果与CSV复核表，保存在 `private_work` 中。各结果文件记录的语料哈希、题集哈希与Top-5规则一致；这支持它们声明使用相同输入，但不等于事后重新核实了云端实际加载的全部输入内容。

本次实际批量大小始终为4。若复现者遇到显存不足，可以只把索引格的 `--batch-size` 改为 `1`；这是备用处理方法，本次记录没有显示发生过显存不足或实际使用过该降级。

## 9. 实际成绩：不同指标呈现不同结果

下面的数值从下载的 `AUP_results_private.zip` 逐题重算，已检查排名、页码命中、原句命中和汇总值是否一致。公开摘要见[运行摘要](results/run_summary.json)及[逐题指标](results/per_question_metrics.csv)。

这属于**既有导出记录的内部一致性复算**：用结果中的候选文本核对其记录的参考原句，并重新计算指标。结果ZIP不包含原始`chunks.jsonl`、`questions.jsonl`、向量文件或云端最终脚本，复算没有重新计算原始语料/题集的文件哈希，没有重跑模型排序，也没有阅读PDF独立确认标注或科学解释。

### 15道测试题

| 方法 | 目标页命中@5 | 严格原句命中@5 | 页码MRR@5 |
|---|---:|---:|---:|
| BM25 | 10/15（66.7%） | 7/15（46.7%） | 0.4356 |
| Qwen语义检索 | 10/15（66.7%） | 5/15（33.3%） | 0.5389 |

### 5道开发题

| 方法 | 目标页命中@5 | 严格原句命中@5 | 页码MRR@5 |
|---|---:|---:|---:|
| BM25 | 3/5（60%） | 2/5（40%） | 0.3500 |
| Qwen语义检索 | 4/5（80%） | 2/5（40%） | 0.6000 |

三个指标的含义不同：

- **目标页命中@5：** 前5个候选片段中，至少一个位于题目指定论文的指定物理页。
- **严格原句命中@5：** 在指定页的候选片段中，包含题目参考原句；检查前仅统一空白字符。
- **页码MRR@5：** 首次找到目标页的排名为r，该题计1/r；没找到计0，再对全部问题取平均。

因此，页码找对但具体段落没找对时，会出现“页码命中、原句未命中”。几道实际测试题就体现了这种差异：

| 题目ID与主题 | BM25 | 语义检索 | 能说明什么 |
|---|---|---|---|
| T09：冷藏后复苏进水盐度 | 未命中目标页 | 第1名命中页与原句 | 语义检索补到了关键词方法遗漏的一个出处 |
| T03：每日运行周期与沉淀时间 | 第2名命中页与原句 | 前5名未命中目标页 | 关键词方法也有自己的命中优势 |
| T01：快速颗粒化中额外盐投加条件 | 未命中目标页 | 第2名命中页，未命中严格原句 | 找到目标页还不足以确认问题已得到回答 |

这里不对候选段落的科学解释作额外判定；判断盐种、剂量、单位和适用条件，还要回到完整原文。

这组题也有明确局限：只有10篇论文，问题集中于方法和运行条件，参考位置全部在PDF物理第2页；开发题和测试题按问题划分，来源论文有重叠。参考标注仍待独立人工确认，导出CSV的人工复核栏也尚为空。因此，这些成绩是小范围证据定位检查，不能当作通用问答准确率、跨文献泛化能力或科研结论正确率。

## 10. 保存结果：运行完成与Notebook保存完整是两件事

完成评测后，我下载了 `AUP_results_private.zip` 和 `01_AUP_GPU_run.ipynb`。

这次整理时发现一个留档细节：结果ZIP包含预定的设备、最小推理、索引元数据和四组评测文件，但下载的Notebook只保留了部分执行输出，其中索引格停留在68%的进度显示，后续评测、汇总与打包格子没有最终输出。Notebook共9格代码，只有初始化、GPU检查、输入核验和最小模型推理这4格有执行编号。

这些文件支持“导出了运行结果记录，成绩复算一致”的结论；不能把这个Notebook称为完整执行日志，也不能将“结果文件齐全”写成“全部步骤已独立复现成功”。结果包还缺少向量文件、云端最终源码及其哈希、修复后的模型清单和完整命令日志。本次没有重新启动AUP补跑，也没有补写缺失的历史输出。

以后运行到最后，先等待单元格结束，按 **Ctrl+S**，再下载Notebook并在本地检查末尾输出。结果包与Notebook各留一份：前者保存结构化记录，后者保存代码与交互输出。若需要逐条追溯排错，还应另存终端日志。

本次结果包的主要内容是：

```text
private_work/hardware.json                  设备与实际张量检查
private_work/model_smoke.json               最小模型推理
private_work/index_metadata.json            索引参数、语料哈希与计时
private_work/preparation_validation.json    输入和参考原句检查
private_work/requirements-lock.txt          环境版本
private_work/evaluation_{bm25,dense}_{dev,test}.{json,csv}
run_config.json
questions_version.json
```

持有本次原始私有结果包时，可以运行本分享的核验脚本。该脚本限定2026年10月24日的固定归档，先检查整个ZIP的SHA256；其他归档会被拒绝，避免把固定运行日期和数据局限套用到新的实验。下面是在 **AUP Terminal** 中的示例：

```bash
python3 scripts/summarize_results.py /path/to/AUP_results_private.zip --out results
```

这里把 `/path/to/` 替换为文件实际位置。已配置本地Python命令的读者，也可在分享目录打开 **Windows PowerShell** 执行：

```powershell
python scripts/summarize_results.py "$env:USERPROFILE\Downloads\AUP_results_private.zip" --out results
```

它仅使用Python标准库，逐题重算指标，比较结果文件中记录的模型版本、语料哈希、题集哈希和Top-5规则，按字段白名单输出本次数值摘要。它不重新计算原始输入文件哈希、不重新调用GPU，也不是通用脱敏工具；本次公开输出另做了敏感信息检查。

从分享代码新建Notebook、使用自己文献的读者，也可在计算完成后回到Terminal，进入项目目录。按所走分支选择下面一条打包命令即可。

只做单题检索、没有建立题集：

```bash
cd /home/jovyan/aup_evidence_project
python3 -m zipfile -c AUP_my_results.zip private_work run_config.json
```

已经准备 `questions.jsonl` 并完成评测：

```bash
cd /home/jovyan/aup_evidence_project
python3 -m zipfile -c AUP_my_results.zip private_work run_config.json questions.jsonl
```

在左侧文件栏刷新后，右键 `AUP_my_results.zip` → **Download**；Notebook按Ctrl+S后也右键下载。这个包包含自己的原文和问题，仅用于个人留档。本次专用的 `summarize_results.py` 要求原课程结果包的核验与版本文件，不能直接用于缺少这些文件的自备数据包；自备数据的数值可查看各 `evaluation_*.json`。

文件下载完成后，返回AUP控制页点击 **Stop my server**。关闭浏览器标签页不一定会停止服务器。[停止服务器说明](https://amdresearch.github.io/aup-learning-cloud/user-guide/platform-basics.html#stop-your-server)

## 11. 我会保留的几条经验

1. **先把数据位置确定下来。** 当前工作目录、Jupyter文件栏目录和持久化目录需要分别确认。
2. **把文件准备与GPU计算拆开考虑。** 大文件上传可能比推理更耗时，云端直下模型能绕开本地上行链路，但仍依赖服务器外网连接。
3. **改变下载方式时，检查项目附加文件。** 官方权重齐全，不代表本项目要求的清单也已齐全。
4. **安装完成后测试当前内核。** `import`、解释器位置和包版本，比只看安装日志更直接。
5. **每个阶段留下可检查的产物。** 设备检查、最小推理、索引、评测逐步确认，失败时更容易定位。
6. **同时看定位、原句和人工复核。** 一个更高的排序指标不能替代对具体实验条件的核对。

ChatGPT在这次实践中主要帮助识别界面、解释日志和排查错误；Codex负责准备代码、输入与本次分享整理。实际运行和文件下载由我完成，科研判断和原文复核仍由我负责。模型版本、原文出处和真实结果文件，使这些协作过程能够接受检查。

如果继续做下去，我会先完成15题的独立原文复核，记录哪些候选段落真正支持问题，再考虑扩大问题范围。这一步可以直接判断这个工具是否帮我减少了查找时间，以及是否遗漏了影响科研解释的重要条件。
