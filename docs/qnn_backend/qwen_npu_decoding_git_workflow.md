# QNN Decoding 功能开发 - Git 工作流指南

本文档提供 QNN Decoding 功能开发的完整 Git 工作流，从创建功能分支到提交 PR 的每一步都有详细说明。

## 📋 目录

1. [前置准备](#前置准备)
2. [阶段 1: 创建功能分支](#阶段-1-创建功能分支)
3. [阶段 2: 日常开发流程](#阶段-2-日常开发流程)
4. [阶段 3: 提交和推送](#阶段-3-提交和推送)
5. [阶段 4: 创建 Pull Request](#阶段-4-创建-pull-request)
6. [常见问题](#常见问题)
7. [快速参考命令](#快速参考命令)

---

## 前置准备

### 1. 检查 Git 配置

确保 Git 已配置用户信息：

```bash
# 检查当前配置
git config user.name
git config user.email

# 如果未配置，设置全局配置
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

### 2. 检查远程仓库配置

```bash
# 查看远程仓库
git remote -v
```

**期望输出：**
```
origin  git@github.com:jialilve/mllm.git (fetch)
origin  git@github.com:jialilve/mllm.git (push)
upstream        https://github.com/UbiquitousLearning/mllm.git (fetch)
upstream        https://github.com/UbiquitousLearning/mllm.git (push)
```

**如果没有 upstream，添加它：**
```bash
git remote add upstream https://github.com/UbiquitousLearning/mllm.git
```

### 3. 检查当前状态

```bash
# 查看当前分支
git branch

# 查看当前状态
git status

# 查看最近的提交历史
git log --oneline -5
```

---

## 阶段 1: 创建功能分支

### 步骤 1.1: 同步 upstream 最新代码

在创建功能分支之前，确保基于最新的 upstream/v2 代码：

```bash
# 1. 获取 upstream 的最新更改
git fetch upstream

# 2. 查看 upstream/v2 和本地 v2 的差异（可选）
git log v2..upstream/v2 --oneline

# 3. 如果 upstream 有更新，同步到本地 v2（可选，用于保持本地 v2 最新）
git checkout v2
git merge upstream/v2
# 或者使用 rebase（更推荐，保持提交历史整洁）
# git rebase upstream/v2
```

### 步骤 1.2: 创建功能分支

**重要：** 功能分支应该基于 `upstream/v2` 创建，而不是 `origin/v2` 或本地 `v2`。

```bash
# 创建并切换到新功能分支
git checkout -b feature/qwen-npu-decoding upstream/v2
```

**分支命名规范：**
- `feature/` - 新功能
- `fix/` - 修复 bug
- `refactor/` - 重构
- `docs/` - 文档更新

**示例：**
- ✅ `feature/qwen-npu-decoding` - 新功能
- ✅ `fix/qnn-kv-cache-sync` - 修复
- ❌ `my-branch` - 不推荐，不够描述性

### 步骤 1.3: 验证分支状态

```bash
# 确认当前在功能分支上
git branch

# 应该显示 * feature/qwen-npu-decoding

# 查看分支基于哪个提交
git log --oneline -1

# 查看与 upstream/v2 的关系
git log --oneline --graph --decorate -5
```

---

## 阶段 2: 日常开发流程

### 2.1 开始开发

在功能分支上进行开发：

```bash
# 确认在功能分支上
git branch

# 开始编辑文件、添加代码等
# ...
```

### 2.2 查看修改状态

定期检查你的修改：

```bash
# 查看哪些文件被修改
git status

# 查看具体的修改内容
git diff

# 查看某个文件的修改
git diff <文件路径>

# 查看已暂存和未暂存的修改
git diff --staged  # 已暂存
git diff          # 未暂存
```

### 2.3 暂存修改（准备提交）

```bash
# 暂存所有修改
git add .

# 或者暂存特定文件
git add <文件路径1> <文件路径2>

# 或者暂存特定目录
git add <目录路径>/

# 查看暂存的文件
git status
```

**最佳实践：**
- 相关修改一起提交（例如：接口定义和实现一起提交）
- 不相关的修改分开提交
- 每次提交应该是一个逻辑完整的改动

### 2.4 提交修改

```bash
# 提交暂存的修改
git commit -m "提交信息"
```

**提交信息规范：**

格式：`<类型>: <简短描述>`

**类型：**
- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档更新
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具相关

**示例：**

```bash
# 好的提交信息
git commit -m "feat: add KV cache interface extension for Qwen NPU decoding"
git commit -m "fix: correct position_ids handling in decode loop"
git commit -m "docs: update decoding requirements document"

# 多行提交信息（推荐用于复杂改动）
git commit -m "feat: implement decoding loop for Qwen NPU

- Add KV cache sequence count management
- Implement decode loop with position_ids handling
- Add EOS token termination check
- Update forward method to support decode phase"
```

**不好的提交信息：**
```bash
# ❌ 太简单
git commit -m "update"

# ❌ 不够描述性
git commit -m "fix bug"

# ❌ 使用中文（除非项目要求）
git commit -m "修复问题"
```

---

## 阶段 3: 提交和推送

### 3.1 提交到本地仓库

```bash
# 提交修改
git add .
git commit -m "feat: your commit message"
```

### 3.2 推送到 Fork（origin）

**第一次推送：**

```bash
# 推送功能分支到 origin（你的 Fork）
git push -u origin feature/qwen-npu-decoding
```

`-u` 参数设置上游分支，之后可以直接使用 `git push`。

**后续推送：**

```bash
# 如果已设置上游分支
git push

# 或者明确指定
git push origin feature/qwen-npu-decoding
```

### 3.3 处理推送冲突

如果 upstream 有更新，你的分支可能落后：

```bash
# 1. 获取最新代码
git fetch upstream

# 2. 在功能分支上 rebase upstream/v2
git rebase upstream/v2

# 3. 如果有冲突，解决冲突后继续
# 解决冲突后：
git add <冲突文件>
git rebase --continue

# 4. 如果 rebase 过程中想取消
git rebase --abort

# 5. 强制推送（因为 rebase 改变了历史）
git push --force-with-lease origin feature/qwen-npu-decoding
```

**注意：** 使用 `--force-with-lease` 比 `--force` 更安全，它会检查远程分支是否有其他人的提交。

---

## 阶段 4: 创建 Pull Request

### 4.1 推送功能分支

确保所有修改都已提交并推送：

```bash
# 检查状态
git status

# 如果有未提交的修改，先提交
git add .
git commit -m "feat: final changes"

# 推送到 Fork
git push origin feature/qwen-npu-decoding
```

### 4.2 在 GitHub 上创建 PR

#### 方法 1: 通过 GitHub Web 界面

1. **访问你的 Fork 仓库：**
   ```
   https://github.com/jialilve/mllm
   ```

2. **你会看到提示创建 PR：**
   - GitHub 通常会在你推送新分支后显示提示
   - 点击 "Compare & pull request" 按钮

3. **或者手动创建：**
   - 点击 "Pull requests" 标签
   - 点击 "New pull request"
   - 选择：
     - **base repository:** `UbiquitousLearning/mllm`
     - **base branch:** `v2`
     - **compare repository:** `jialilve/mllm`
     - **compare branch:** `feature/qwen-npu-decoding`

#### 方法 2: 使用 GitHub CLI（如果已安装）

```bash
# 创建 PR
gh pr create --base v2 --head jialilve:feature/qwen-npu-decoding --title "feat: Qwen NPU Decoding Support" --body "PR描述内容"
```

### 4.3 编写 PR 描述

**PR 标题格式：**
```
feat: Qwen NPU Decoding Support
```

**PR 描述模板：**

```markdown
## 功能描述
实现 Qwen NPU 自回归解码功能，支持连续 token 生成。

## 主要改动
- 扩展 KV Cache 接口，支持序列长度管理
- 实现解码循环，支持 position_ids 自动递增
- 添加 EOS token 终止检查
- 更新 forward 方法以支持 decode 阶段

## 实现细节
- 在 `QwenForCausalLM` 中添加 `setKVCacheSeqCnt` 方法
- 实现基于 128 长度 KV cache 的解码循环
- 正确处理 position_ids 的传递和递增

## 测试
- [x] 编译通过
- [x] 单次 prefill 测试通过
- [x] 解码循环测试通过
- [x] EOS token 终止测试通过

## 相关文档
- [需求文档](../docs/qnn_backend/qwen_npu_decoding_requirements.md)

## 相关 Issue
#<issue_number> (如果有)
```

### 4.4 PR 提交清单

在创建 PR 之前，确认：

- [ ] 代码已编译通过，无编译错误
- [ ] 已运行相关测试，测试通过
- [ ] 代码已格式化（如果有格式化工具）
- [ ] 提交信息清晰，符合规范
- [ ] 所有修改都已提交并推送
- [ ] PR 描述清晰，说明了功能和改动
- [ ] 已同步 upstream/v2 最新代码（避免冲突）

---

## 常见问题

### Q1: 如何查看功能分支和 upstream/v2 的差异？

```bash
# 查看所有差异
git diff upstream/v2..feature/qwen-npu-decoding

# 查看提交历史差异
git log upstream/v2..feature/qwen-npu-decoding --oneline

# 查看文件列表差异
git diff --name-only upstream/v2..feature/qwen-npu-decoding
```

### Q2: 如何修改已提交的 commit？

**修改最后一次提交：**

```bash
# 修改提交信息
git commit --amend -m "新的提交信息"

# 添加遗漏的文件到上次提交
git add <遗漏的文件>
git commit --amend --no-edit

# 修改后需要强制推送
git push --force-with-lease origin feature/qwen-npu-decoding
```

**修改更早的提交：**

```bash
# 使用交互式 rebase
git rebase -i HEAD~3  # 修改最近 3 个提交

# 在编辑器中，将需要修改的提交标记为 'edit'
# 然后修改文件，执行：
git add .
git commit --amend
git rebase --continue
```

### Q3: 如何撤销未提交的修改？

```bash
# 撤销工作区的修改（未暂存）
git checkout -- <文件路径>
# 或者
git restore <文件路径>

# 撤销所有未暂存的修改
git checkout -- .
# 或者
git restore .

# 撤销暂存的修改（但保留工作区修改）
git reset HEAD <文件路径>
# 或者
git restore --staged <文件路径>
```

### Q4: 如何查看分支的提交历史？

```bash
# 简洁模式
git log --oneline

# 图形化显示
git log --oneline --graph --decorate

# 显示最近 10 个提交
git log --oneline -10

# 显示某个文件的提交历史
git log --oneline <文件路径>
```

### Q5: 如何切换分支？

```bash
# 切换到其他分支
git checkout <分支名>

# 或者使用新的命令（Git 2.23+）
git switch <分支名>

# 创建并切换新分支
git checkout -b <新分支名>
# 或者
git switch -c <新分支名>
```

### Q6: 如何删除分支？

```bash
# 删除本地分支
git branch -d feature/qwen-npu-decoding

# 强制删除本地分支（即使未合并）
git branch -D feature/qwen-npu-decoding

# 删除远程分支
git push origin --delete feature/qwen-npu-decoding
```

### Q7: PR 被要求修改后怎么办？

```bash
# 1. 在功能分支上继续修改
git checkout feature/qwen-npu-decoding

# 2. 进行修改
# ... 编辑文件 ...

# 3. 提交修改
git add .
git commit -m "fix: address review comments"

# 4. 推送到 Fork
git push origin feature/qwen-npu-decoding

# PR 会自动更新，不需要重新创建
```

### Q8: 如何同步 upstream 的最新代码到功能分支？

```bash
# 方法 1: 使用 rebase（推荐，保持提交历史整洁）
git fetch upstream
git rebase upstream/v2

# 如果有冲突，解决后：
git add <冲突文件>
git rebase --continue

# 方法 2: 使用 merge
git fetch upstream
git merge upstream/v2
```

---

## 快速参考命令

### 日常开发流程

```bash
# 1. 切换到功能分支
git checkout feature/qwen-npu-decoding

# 2. 查看状态
git status

# 3. 暂存修改
git add .

# 4. 提交
git commit -m "feat: your message"

# 5. 推送
git push
```

### 创建功能分支（一次性）

```bash
# 1. 同步 upstream
git fetch upstream

# 2. 创建功能分支
git checkout -b feature/qwen-npu-decoding upstream/v2

# 3. 推送并设置上游
git push -u origin feature/qwen-npu-decoding
```

### 同步 upstream 代码

```bash
# 1. 获取最新代码
git fetch upstream

# 2. 在功能分支上 rebase
git checkout feature/qwen-npu-decoding
git rebase upstream/v2

# 3. 如果有冲突，解决后继续
git add <冲突文件>
git rebase --continue

# 4. 强制推送
git push --force-with-lease
```

### 查看差异和状态

```bash
# 查看工作区修改
git diff

# 查看与 upstream/v2 的差异
git diff upstream/v2..feature/qwen-npu-decoding

# 查看提交历史
git log --oneline --graph --decorate -10
```

---

## 完整工作流示例

假设你要实现 QNN Decoding 功能，完整流程如下：

```bash
# ========== 阶段 1: 创建功能分支 ==========

# 1. 同步 upstream
git fetch upstream

# 2. 创建功能分支
git checkout -b feature/qwen-npu-decoding upstream/v2

# 3. 推送并设置上游
git push -u origin feature/qwen-npu-decoding


# ========== 阶段 2: 开发 ==========

# 1. 开始开发（编辑文件）
vim mllm/models/qwen_npu/modeling_qwen_npu.hpp
# ... 添加代码 ...

# 2. 查看修改
git status
git diff

# 3. 暂存并提交
git add mllm/models/qwen_npu/modeling_qwen_npu.hpp
git commit -m "feat: add KV cache interface extension"

# 4. 继续开发
vim mllm/models/qwen_npu/modeling_qwen_npu.cpp
# ... 添加代码 ...

# 5. 再次提交
git add mllm/models/qwen_npu/modeling_qwen_npu.cpp
git commit -m "feat: implement setKVCacheSeqCnt method"

# 6. 定期推送
git push


# ========== 阶段 3: 准备 PR ==========

# 1. 确保所有修改已提交
git status

# 2. 同步 upstream（避免冲突）
git fetch upstream
git rebase upstream/v2

# 3. 如果有冲突，解决后继续
# git add <冲突文件>
# git rebase --continue

# 4. 强制推送（如果 rebase 了）
git push --force-with-lease

# 5. 在 GitHub 上创建 PR
# 访问: https://github.com/jialilve/mllm
# 点击 "Compare & pull request"
```

---

## 总结

**标准工作流：**

1. ✅ **创建功能分支** - 基于 `upstream/v2`
2. ✅ **开发** - 在功能分支上编辑、提交
3. ✅ **推送** - 定期推送到 Fork
4. ✅ **同步** - 必要时同步 upstream 代码
5. ✅ **PR** - 在 GitHub 上创建 Pull Request

**关键原则：**

- 🎯 每个功能使用独立分支
- 🎯 功能分支基于 `upstream/v2`
- 🎯 提交信息清晰、规范
- 🎯 定期推送，避免丢失工作
- 🎯 PR 前同步 upstream，避免冲突

---

**需要帮助？** 如果遇到问题，可以：
- 查看本文档的"常见问题"部分
- 使用 `git help <命令>` 查看帮助
- 参考项目的其他 PR 示例

