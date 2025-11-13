# QNN Execute Return Order Fix - PR 准备指南

## 问题 1: 调试日志信息处理

### 当前情况
代码中添加了很多调试用的 `MLLM_INFO` 打印信息，用于验证 QNN Execute Return Order 修复是否正确。

### 解决方案

#### 方案 A: 使用条件编译宏（推荐）
创建一个调试宏，可以通过编译选项控制是否启用调试日志：

**优点：**
- 代码保持整洁，不需要注释/取消注释
- 可以通过编译选项控制（如 `-DMLLM_QNN_DEBUG_OUTPUT_ORDER=ON`）
- 保留所有调试代码，方便将来使用

**实现步骤：**
1. 在 `QNNBackend.hpp` 或相关头文件中定义宏
2. 将所有调试日志用宏包裹
3. 在 CMakeLists.txt 中添加编译选项

#### 方案 B: 注释掉调试日志（简单快速）
直接注释掉所有调试用的 `MLLM_INFO`，但保留代码以便将来使用。

**优点：**
- 实现简单快速
- 代码清晰，明确标注为调试代码
- 需要时可以快速取消注释

**缺点：**
- 代码中会有很多注释，可能不够美观
- 需要手动注释/取消注释

#### 方案 C: 使用日志级别控制
利用现有的 `LogLevel` 机制，将调试日志改为 `MLLM_DEBUG`（如果存在）或通过设置日志级别控制。

**注意：** 当前代码库中似乎没有 `MLLM_DEBUG`，只有 `MLLM_INFO/WARN/ERROR`。

### 推荐方案
**建议使用方案 B（注释）**，因为：
1. 实现最简单，不需要修改构建系统
2. 代码意图清晰，明确标注为调试代码
3. 需要时可以快速恢复
4. 对于 PR 来说，注释掉的调试代码是可以接受的

### 需要处理的文件
- `mllm/backends/qnn/QNNBackend.cpp` - 包含大部分调试日志
- `mllm/models/qwen_npu/modeling_qwen_npu.hpp` - 已注释（保持现状）

---

## 问题 2: Git 工作流同步

### 当前情况
- Fork 的项目与原项目（upstream）没有同步
- 本地仓库与 Fork 的项目也没有同步
- **重要：** 当前在 fork 的 `v2` 分支上工作，需要 PR 到主项目的 `v2` 分支
- **重要：** fork 的 `v2` 分支上有多个 commits，但只有最新的这次修改需要 PR

### 正确的 Git 工作流策略

#### 为什么需要功能分支？

**最佳实践：** 每个功能/修复应该创建独立的功能分支，而不是直接在主分支（如 `v2`）上工作。

**优点：**
- ✅ 可以独立 PR 每个功能，不需要一次性 PR 所有改动
- ✅ 保持主分支干净，只包含已合并的功能
- ✅ 方便代码审查，每个 PR 只关注一个功能
- ✅ 如果某个功能有问题，不影响其他功能

#### 当前情况的解决方案

如果你已经在 `v2` 分支上做了多个 commits，但只想 PR 其中一个，有以下几种方案：

**方案 A: 创建新功能分支并 cherry-pick（推荐）**

这是最干净的方法，创建一个新的功能分支，只包含你需要的修改：

```bash
# 1. 确保 upstream 已配置并同步
git fetch upstream
git checkout v2
git merge upstream/v2  # 或 git rebase upstream/v2

# 2. 创建一个新的功能分支，基于 upstream/v2
git checkout -b fix/qnn-execute-return-order upstream/v2

# 3. 找到你需要 PR 的 commit（假设是最新的 commit）
# 查看最近的 commits
git log --oneline -10

# 4. Cherry-pick 你需要的 commit(s)
# 如果是最新的 commit
git cherry-pick HEAD@{1}  # 或者使用 commit hash
# 或者如果是多个相关的 commits
git cherry-pick <commit1> <commit2> ...

# 5. 如果有未提交的更改，先提交
git add .
git commit -m "fix: QNN Execute Return Order - handle output reordering"

# 6. 推送到你的 Fork
git push origin fix/qnn-execute-return-order

# 7. 在 GitHub 上创建 PR：从 fix/qnn-execute-return-order 到 upstream/v2
```

**方案 B: 使用交互式 rebase 整理 commits**

如果你想保留在 `v2` 分支上工作，但只 PR 部分 commits：

```bash
# 1. 创建一个新分支用于 PR
git checkout -b fix/qnn-execute-return-order

# 2. 使用交互式 rebase 整理 commits
git rebase -i upstream/v2

# 在编辑器中，只保留需要 PR 的 commits，其他标记为 drop
# 或者使用 squash 合并多个相关 commits

# 3. 推送到 Fork
git push origin fix/qnn-execute-return-order
```

**方案 C: 创建补丁并应用到新分支**

```bash
# 1. 在 v2 分支上，创建补丁文件
git format-patch -1 HEAD  # 为最新的 commit 创建补丁

# 2. 创建新功能分支
git checkout -b fix/qnn-execute-return-order upstream/v2

# 3. 应用补丁
git am <patch-file>

# 4. 推送到 Fork
git push origin fix/qnn-execute-return-order
```

### 正确的 Git 工作流（未来参考）

#### 步骤 1: 配置远程仓库
```bash
# 查看当前远程仓库
git remote -v

# 如果没有 upstream，添加原项目为 upstream
git remote add upstream <原项目URL>

# 如果已有 upstream，确认 URL 正确
git remote set-url upstream <原项目URL>
```

#### 步骤 2: 同步 upstream 到本地（v2 分支）
```bash
# 获取 upstream 的最新更改
git fetch upstream

# 切换到 v2 分支
git checkout v2

# 合并 upstream 的更改到本地 v2 分支
git merge upstream/v2

# 或者使用 rebase（更推荐，保持提交历史整洁）
git rebase upstream/v2
```

#### 步骤 3: 同步本地 v2 分支到 Fork（可选）
```bash
# 推送本地 v2 分支到你的 Fork（用于同步，不是 PR）
git push origin v2
```

#### 步骤 4: 创建功能分支（从当前 v2 分支提取需要的修改）
```bash
# 方法 1: 如果修改还未提交，直接创建新分支
git checkout -b fix/qnn-execute-return-order upstream/v2
# 然后手动应用你的修改，或使用 git cherry-pick

# 方法 2: 如果修改已提交，使用 cherry-pick
# 先找到你的 commit hash
git log --oneline -10

# 创建新分支基于 upstream/v2
git checkout -b fix/qnn-execute-return-order upstream/v2

# Cherry-pick 你需要的 commit(s)
git cherry-pick <commit-hash>

# 方法 3: 如果修改还未提交，先暂存
git stash
git checkout -b fix/qnn-execute-return-order upstream/v2
git stash pop
# 然后提交
```

#### 步骤 5: 处理调试日志并提交
```bash
# 1. 注释掉所有调试日志（使用方案 B）
# 2. 确保代码编译通过
# 3. 运行测试确保功能正常

# 提交更改
git add .
git commit -m "fix: QNN Execute Return Order - handle output reordering

- Fix QNN graphExecute output order mismatch
- Add output reordering logic based on expected order
- Remove debug logs for production (commented for future use)"
```

#### 步骤 6: 重新编译和测试
```bash
# 清理之前的构建
rm -rf build/

# 重新编译
# 根据你的构建系统执行编译命令
# 例如：cmake .. && make

# 运行测试
# 确保所有测试通过
```

#### 步骤 7: 推送到 Fork 并创建 PR
```bash
# 推送功能分支到你的 Fork
git push origin fix/qnn-execute-return-order

# 在 GitHub 上创建 Pull Request
# 从你的 Fork: fix/qnn-execute-return-order
# 到原项目: v2 分支
```

### 针对当前情况的快速操作指南

如果你现在在 `v2` 分支上，有未提交的修改或已提交的修改，按以下步骤操作：

#### 情况 1: 修改还未提交
```bash
# 1. 暂存当前修改
git stash

# 2. 同步 upstream/v2
git fetch upstream
git checkout v2
git rebase upstream/v2

# 3. 创建功能分支
git checkout -b fix/qnn-execute-return-order upstream/v2

# 4. 应用你的修改
git stash pop

# 5. 提交修改
git add .
git commit -m "fix: QNN Execute Return Order - handle output reordering

- Fix QNN graphExecute output order mismatch
- Add output reordering logic based on expected order
- Remove debug logs for production (commented for future use)"

# 6. 推送到 Fork
git push origin fix/qnn-execute-return-order
```

#### 情况 2: 修改已提交（在 v2 分支上）
```bash
# 1. 查看最近的 commits，找到你的 commit hash
git log --oneline -10

# 2. 同步 upstream/v2
git fetch upstream
git checkout v2
git rebase upstream/v2

# 3. 创建功能分支
git checkout -b fix/qnn-execute-return-order upstream/v2

# 4. Cherry-pick 你的 commit(s)
# 假设你的 commit hash 是 abc1234
git cherry-pick abc1234

# 如果有多个相关 commits，可以一起 cherry-pick
# git cherry-pick abc1234 def5678

# 5. 推送到 Fork
git push origin fix/qnn-execute-return-order
```

#### 情况 3: 有多个 commits，但只想 PR 最新的
```bash
# 1. 查看 commits，确认哪些需要 PR
git log --oneline -10

# 2. 同步 upstream/v2
git fetch upstream
git checkout v2
git rebase upstream/v2

# 3. 创建功能分支
git checkout -b fix/qnn-execute-return-order upstream/v2

# 4. Cherry-pick 最新的 commit（或相关的几个 commits）
git cherry-pick HEAD@{1}  # 或者使用具体的 commit hash

# 5. 推送到 Fork
git push origin fix/qnn-execute-return-order
```

### 关于功能分支的常见问题

**Q: 必须全部一起 PR 到主项目吗？**
A: **不是的！** 这正是为什么需要功能分支的原因。每个功能分支可以独立 PR，不需要一次性 PR 所有改动。

**Q: 我的开发流程有问题吗？**
A: 在 `v2` 分支上直接开发是可以的（特别是如果你在 fork 上工作），但更好的做法是：
- 为每个功能创建独立的功能分支
- 功能分支基于 `upstream/v2` 创建
- 只将需要的功能分支 PR 到主项目
- 其他不需要 PR 的改动保留在你的 fork 分支上

**Q: 我没创建过功能分支，怎么办？**
A: 不用担心！创建功能分支很简单：
```bash
# 创建新分支
git checkout -b fix/qnn-execute-return-order upstream/v2

# 或者从当前分支创建
git checkout -b fix/qnn-execute-return-order
```
功能分支就是普通的 Git 分支，可以随时创建、删除、合并。

**Q: 如果我在 v2 分支上有很多 commits，只想 PR 其中一个怎么办？**
A: 使用 `cherry-pick`：
```bash
# 1. 创建新功能分支
git checkout -b fix/qnn-execute-return-order upstream/v2

# 2. Cherry-pick 你需要的 commit
git cherry-pick <commit-hash>

# 3. 推送到 Fork 并创建 PR
git push origin fix/qnn-execute-return-order
```

---

## 前置条件：Git 配置

**重要：** 在开始之前，确保 Git 已配置 `user.name` 和 `user.email`。

如果遇到 "Committer identity unknown" 错误，请先配置 Git：

```bash
# 全局配置（推荐）
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

详细说明请参考：[Git 配置说明](./GIT_SETUP.md)

---

## 快速参考：当前情况的操作步骤

### 场景：在 fork 的 v2 分支上有多个 commits，只想 PR 最新的修改

**最简单的方法（推荐）：**

```bash
# 1. 确保 upstream 已配置
git remote add upstream <原项目URL>  # 如果还没有

# 2. 同步 upstream/v2
git fetch upstream

# 3. 查看你的 commits，找到需要 PR 的 commit hash
git log --oneline -10

# 4. 创建功能分支（基于 upstream/v2）
git checkout -b fix/qnn-execute-return-order upstream/v2

# 5. Cherry-pick 你需要的 commit（假设是 abc1234）
git cherry-pick abc1234

# 6. 确保调试日志已注释（已完成）
# 7. 编译和测试
# 8. 推送到 Fork
git push origin fix/qnn-execute-return-order

# 9. 在 GitHub 上创建 PR：从 fix/qnn-execute-return-order 到 upstream/v2
```

**或者使用脚本：**

```bash
# 运行自动化脚本
./docs/qnn_fix_bug/sync_and_prepare_pr.sh

# 脚本会引导你完成所有步骤
```

---

## PR 提交清单

在提交 PR 之前，请确认：

- [ ] 已同步 upstream 和本地仓库
- [ ] 已注释掉所有调试日志（或使用条件编译）
- [ ] 代码已重新编译，无编译错误
- [ ] 已运行测试，所有测试通过
- [ ] 提交信息清晰，描述了修复的问题
- [ ] 代码已推送到 Fork 的功能分支
- [ ] PR 描述清晰，说明了问题和解决方案

---

## PR 描述模板

```markdown
## 问题描述
修复 QNN Execute Return Order 问题：QNN graphExecute 返回的输出顺序与 MLLM 期望的顺序不一致。

## 解决方案
- 在 `QNNBackend::graphExecute` 中添加输出重排序逻辑
- 根据 `expectedOrder` 将 QNN 返回的输出重新排序到 MLLM 期望的顺序
- 添加输出索引映射机制，确保正确匹配 tensor 名称
- 注释掉调试日志，保留代码以便将来调试使用

## 修改的文件
- `mllm/backends/qnn/QNNBackend.cpp` - 添加输出重排序逻辑
- `mllm/backends/qnn/QNNModel.cpp` - 添加输出索引映射方法
- `mllm/backends/qnn/QNNModel.hpp` - 添加输出索引映射方法声明
- `mllm/backends/qnn/passes/QNNGraphBuildPass.cpp` - 设置期望输出顺序

## 测试
- [x] 编译通过
- [x] 运行测试通过
- [x] 验证输出顺序正确

## 相关 Issue
#<issue_number> (如果有)
```

**注意：** PR 的目标分支应该是 `v2`，不是 `main` 或 `master`。

