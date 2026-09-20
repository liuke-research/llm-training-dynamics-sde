# 上传到 GitHub

## 上传前

先解压，进入 `llm-training-dynamics-sde` 文件夹。应能直接看见 `README.md`、`pyproject.toml`、`src`、`tests`。这些文件应位于 GitHub 仓库根目录，而不是把整个 ZIP 当作唯一源码文件提交。

包中保留了公开联系邮箱，请发布前核对。没有包含访问令牌、私有训练语料或模型权重；继续开发时仍需检查自己的新增文件。许可证尚未选择，本次没有代为添加。

## 网页方式

在目标仓库选择 **Add file → Upload files**，把项目根目录中的文件和子文件夹上传并提交。不要漏掉 `.gitignore`、`.gitattributes` 与 `.github`；本地文件管理器可能隐藏点开头文件。

GitHub 官方上传说明：
https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository

该说明当前列出浏览器单文件上限 25 MiB、每次最多 100 个文件。此整理包刻意保留精选实验产物；完整本地运行会产生更多文件，不要连同虚拟环境、缓存和所有检查点一起拖进去。

GitHub 博客中的文件/目录上传演示：
https://github.blog/developer-skills/github/beginners-guide-to-github-uploading-files-and-folders-to-github/

## Git 命令方式（新建且空的远程仓库）

在解压后的项目根目录执行，把 `YOUR_USERNAME` 替换为自己的用户名。已配置 remote 或已有提交的仓库不要照搬初始化命令；不要为了覆盖旧仓库使用强制推送。

```bash
git init
git add .
git commit -m "Add training stability monitor, tiny model and validation"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/llm-training-dynamics-sde.git
git push -u origin main
```

认证使用自己的正常 GitHub 登录方式，不把密码或 token 写入代码文件。

## 上传后的检查

仓库首页应展示 README；能直接浏览 `src/llm_sde/torch_demo.py` 和其他源码；相对文档链接应可打开。CI 文件已经提供，但是否自动运行受仓库权限/设置影响，远程执行结果应以你的 Actions 页面为准。

Actions 配置使用官方 action 的文档接口（本地已检查 YAML 结构，未远程执行）：
https://github.com/actions/checkout
https://github.com/actions/setup-python

此包尚不含 GRPO/PPO 训练，不要把仓库描述改成“已完成 RLHF/GRPO 平台”。可以使用下面的说明：

> Causal training telemetry monitor, tiny Transformer experiments and stochastic-process research baselines. RL post-training integration is planned.
