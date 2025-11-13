modeling_qwen_npu.hpp中只在QwenAttentionProjNPU中的forward函数中最后一行return {query_states, key_states, value_states}
query states在view前的tensor放到return列表的最后 /data/local/tmp/zl/mllm-v2/bin_test目录下的QNNOutputOrderTest输出

```bash
root@zhulei:~/mllm_v2/build-android-qnn-dbg/bin# adb shell
manet:/ $ cd /data/local/tmp/zl/mllm-v2/bin_test
manet:/data/local/tmp/zl/mllm-v2/bin_test $ LD_LIBRARY_PATH=. ./mllm-qwen-npu
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNUtils.cpp:22 QNN Backend Lib: libQnnHtp.so
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:305 Registered Op Package: libQnnLLaMAPackage_CPU.so and interface provider: LLaMAPackageInterfaceProvider
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:305 Registered Op Package: libQnnLLaMAPackage_HTP.so and interface provider: LLaMAPackageInterfaceProvider
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:46 QNN Backend Build Id: v2.36.0.250627101419_123260
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:48 QNN backend supports tensor sparsity
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:51 QNN backend supports dynamic dimensions
[INFO] /root/mllm_v2/mllm/backends/base/PluginSystem.cpp:89 Register customized op: DequantizeAdd:4097 -> QNN
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.0_1' with 3 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.0_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.1_1' with 3 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.1_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.2_1' with 3 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.2_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.3_1' with 3 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.3_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.4_1' with 3 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.4_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.5_1' with 3 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.5_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.6_1' with 3 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.6_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.7_1' with 3 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.7_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.8_1' with 3 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.8_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.9_1' with 3 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.9_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.10_1' with 3 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.10_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.11_1' with 3 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.11_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.12_1' with 3 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.12_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.13_1' with 3 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.13_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.14_1' with 3 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.14_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.15_1' with 3 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.15_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.16_1' with 3 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.16_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.17_1' with 3 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.17_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.18_1' with 3 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.18_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.19_1' with 3 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.19_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.20_1' with 3 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.20_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.21_1' with 3 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.21_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.22_1' with 3 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.22_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.23_1' with 3 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.23_2' with 1 outputs
tensor(
[[151644, 8948, 198, 2610, 525, 264, ..., 30, 151645, 198, 151644, 77091, 198]], dtype=Int64, device=CPU) 
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.0_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1377
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 1378
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 1379
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1377
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 1378
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 1379
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1377) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[1] = QNN[1] (tensor: 1378) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[2] = QNN[2] (tensor: 1379) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.0_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1431
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1431
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1431) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.1_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1451
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 1452
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 1453
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1451
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 1452
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 1453
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1451) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[1] = QNN[1] (tensor: 1452) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[2] = QNN[2] (tensor: 1453) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.1_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1504
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1504
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1504) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.2_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1524
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 1525
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 1526
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1524
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 1525
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 1526
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1524) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[1] = QNN[1] (tensor: 1525) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[2] = QNN[2] (tensor: 1526) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.2_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1577
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1577
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1577) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.3_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1597
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 1598
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 1599
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1597
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 1598
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 1599
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1597) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[1] = QNN[1] (tensor: 1598) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[2] = QNN[2] (tensor: 1599) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.3_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1650
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1650
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1650) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.4_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1670
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 1671
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 1672
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1670
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 1671
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 1672
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1670) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[1] = QNN[1] (tensor: 1671) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[2] = QNN[2] (tensor: 1672) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.4_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1723
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1723
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1723) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.5_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1743
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 1744
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 1745
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1743
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 1744
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 1745
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1743) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[1] = QNN[1] (tensor: 1744) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[2] = QNN[2] (tensor: 1745) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.5_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1796
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1796
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1796) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.6_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1816
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 1817
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 1818
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1816
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 1817
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 1818
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1816) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[1] = QNN[1] (tensor: 1817) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[2] = QNN[2] (tensor: 1818) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.6_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1869
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1869
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1869) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.7_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1889
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 1890
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 1891
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1889
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 1890
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 1891
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1889) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[1] = QNN[1] (tensor: 1890) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[2] = QNN[2] (tensor: 1891) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.7_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1942
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1942
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1942) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.8_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1962
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 1963
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 1964
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1962
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 1963
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 1964
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1962) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[1] = QNN[1] (tensor: 1963) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[2] = QNN[2] (tensor: 1964) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.8_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2015
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2015
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2015) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.9_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2035
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2036
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2037
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2035
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2036
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2037
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2035) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[1] = QNN[1] (tensor: 2036) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[2] = QNN[2] (tensor: 2037) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.9_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2088
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2088
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2088) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.10_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2108
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2109
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2110
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2108
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2109
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2110
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2108) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[1] = QNN[1] (tensor: 2109) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[2] = QNN[2] (tensor: 2110) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.10_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2161
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2161
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2161) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.11_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2181
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2182
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2183
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2181
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2182
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2183
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2181) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[1] = QNN[1] (tensor: 2182) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[2] = QNN[2] (tensor: 2183) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.11_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2234
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2234
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2234) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.12_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2254
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2255
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2256
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2254
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2255
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2256
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2254) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[1] = QNN[1] (tensor: 2255) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[2] = QNN[2] (tensor: 2256) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.12_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2307
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2307
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2307) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.13_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2327
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2328
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2329
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2327
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2328
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2329
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2327) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[1] = QNN[1] (tensor: 2328) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[2] = QNN[2] (tensor: 2329) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.13_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2380
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2380
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2380) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.14_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2400
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2401
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2402
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2400
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2401
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2402
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2400) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[1] = QNN[1] (tensor: 2401) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[2] = QNN[2] (tensor: 2402) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.14_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2453
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2453
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2453) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.15_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2473
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2474
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2475
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2473
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2474
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2475
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2473) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[1] = QNN[1] (tensor: 2474) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[2] = QNN[2] (tensor: 2475) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.15_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2526
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2526
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2526) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.16_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2546
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2547
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2548
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2546
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2547
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2548
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2546) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[1] = QNN[1] (tensor: 2547) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[2] = QNN[2] (tensor: 2548) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.16_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2599
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2599
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2599) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.17_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2619
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2620
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2621
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2619
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2620
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2621
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2619) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[1] = QNN[1] (tensor: 2620) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[2] = QNN[2] (tensor: 2621) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.17_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2672
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2672
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2672) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.18_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2692
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2693
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2694
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2692
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2693
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2694
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2692) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[1] = QNN[1] (tensor: 2693) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[2] = QNN[2] (tensor: 2694) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.18_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2745
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2745
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2745) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.19_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2765
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2766
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2767
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2765
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2766
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2767
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2765) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[1] = QNN[1] (tensor: 2766) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[2] = QNN[2] (tensor: 2767) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.19_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2818
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2818
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2818) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.20_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2838
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2839
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2840
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2838
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2839
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2840
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2838) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[1] = QNN[1] (tensor: 2839) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[2] = QNN[2] (tensor: 2840) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.20_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2891
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2891
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2891) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.21_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2911
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2912
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2913
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2911
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2912
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2913
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2911) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[1] = QNN[1] (tensor: 2912) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[2] = QNN[2] (tensor: 2913) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.21_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2964
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2964
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2964) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.22_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2984
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2985
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2986
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2984
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2985
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2986
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2984) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[1] = QNN[1] (tensor: 2985) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[2] = QNN[2] (tensor: 2986) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.22_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 3037
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 3037
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 3037) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.23_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 3057
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 3058
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 3059
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (3 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 3057
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 3058
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 3059
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 3057) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[1] = QNN[1] (tensor: 3058) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[2] = QNN[2] (tensor: 3059) [SAME]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.23_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 3110
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 3110
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 3110) [SAME]
token: 2121 As
Error: Received signal11 - SIGSEGV (Segmentation violation)
Stack trace:
#0 0x5c8fd2b12c
#1 0x5c8fd2af4c
#2 0x5c8fd2ac38
#3 0x7645b91860 __kernel_rt_sigreturn
#4 0x739fdcbff4
#5 0x739fdad750
#6 0x739fd89748
#7 0x739fb70a74
#8 0x739fb7042c
#9 0x739fafbe14
#10 0x739fafcc68
#11 0x739fafead4
#12 0x764075c3f0 __cxa_finalize
#13 0x764076155c exit
#14 0x7640755158
Possible causes: invalid memory access, dangling pointer, stack overflow.
Shutting down...
```

modeling_qwen_npu.hpp中在QwenAttentionProjNPU中的forward函数中最后一行试试把query states在view前的tensor放到return列表的最后return {query_states, key_states, value_states, query_states_raw};这样修改代码后的/data/local/tmp/zl/mllm-v2/bin_test目录下的QNNOutputOrderTest输出
```bash
manet:/data/local/tmp/zl/mllm-v2/bin_test $ LD_LIBRARY_PATH=. ./mllm-qwen-npu
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNUtils.cpp:22 QNN Backend Lib: libQnnHtp.so
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:305 Registered Op Package: libQnnLLaMAPackage_CPU.so and interface provider: LLaMAPackageInterfaceProvider
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:305 Registered Op Package: libQnnLLaMAPackage_HTP.so and interface provider: LLaMAPackageInterfaceProvider
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:46 QNN Backend Build Id: v2.36.0.250627101419_123260
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:48 QNN backend supports tensor sparsity
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:51 QNN backend supports dynamic dimensions
[INFO] /root/mllm_v2/mllm/backends/base/PluginSystem.cpp:89 Register customized op: DequantizeAdd:4097 -> QNN
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.0_1' with 4 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.0_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.1_1' with 4 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.1_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.2_1' with 4 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.2_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.3_1' with 4 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.3_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.4_1' with 4 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.4_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.5_1' with 4 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.5_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.6_1' with 4 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.6_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.7_1' with 4 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.7_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.8_1' with 4 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.8_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.9_1' with 4 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.9_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.10_1' with 4 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.10_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.11_1' with 4 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.11_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.12_1' with 4 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.12_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.13_1' with 4 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.13_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.14_1' with 4 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.14_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.15_1' with 4 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.15_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.16_1' with 4 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.16_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.17_1' with 4 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.17_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.18_1' with 4 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.18_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.19_1' with 4 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.19_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.20_1' with 4 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.20_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.21_1' with 4 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.21_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.22_1' with 4 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.22_2' with 1 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.23_1' with 4 outputs
[INFO] /root/mllm_v2/mllm/backends/qnn/passes/QNNGraphBuildPass.cpp:185 QNNGraphBuildPass: Recorded MLLM expected output order for graph 'model.layers.23_2' with 1 outputs
tensor(
[[151644, 8948, 198, 2610, 525, 264, ..., 30, 151645, 198, 151644, 77091, 198]], dtype=Int64, device=CPU) 
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.0_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1377
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 1378
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 1379
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [3] 1362
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1362
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 1377
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 1378
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [3] 1379
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:590   [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[0] expects '1377' but it's at QNN[1]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[1] expects '1378' but it's at QNN[2]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[2] expects '1379' but it's at QNN[3]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[3] expects '1362' but it's at QNN[0]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[0] = QNN[1] (tensor: 1377) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[1] = QNN[2] (tensor: 1378) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[2] = QNN[3] (tensor: 1379) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[3] = QNN[0] (tensor: 1362) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.0_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1431
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1431
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1431) [SAME]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.1_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1451
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 1452
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 1453
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [3] 1436
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1436
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 1451
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 1452
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [3] 1453
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:590   [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[0] expects '1451' but it's at QNN[1]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[1] expects '1452' but it's at QNN[2]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[2] expects '1453' but it's at QNN[3]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[3] expects '1436' but it's at QNN[0]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[0] = QNN[1] (tensor: 1451) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[1] = QNN[2] (tensor: 1452) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[2] = QNN[3] (tensor: 1453) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[3] = QNN[0] (tensor: 1436) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.1_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1504
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1504
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1504) [SAME]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.2_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1524
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 1525
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 1526
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [3] 1509
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1509
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 1524
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 1525
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [3] 1526
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:590   [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[0] expects '1524' but it's at QNN[1]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[1] expects '1525' but it's at QNN[2]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[2] expects '1526' but it's at QNN[3]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[3] expects '1509' but it's at QNN[0]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[0] = QNN[1] (tensor: 1524) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[1] = QNN[2] (tensor: 1525) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[2] = QNN[3] (tensor: 1526) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[3] = QNN[0] (tensor: 1509) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.2_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1577
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1577
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1577) [SAME]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.3_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1597
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 1598
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 1599
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [3] 1582
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1582
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 1597
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 1598
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [3] 1599
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:590   [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[0] expects '1597' but it's at QNN[1]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[1] expects '1598' but it's at QNN[2]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[2] expects '1599' but it's at QNN[3]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[3] expects '1582' but it's at QNN[0]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[0] = QNN[1] (tensor: 1597) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[1] = QNN[2] (tensor: 1598) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[2] = QNN[3] (tensor: 1599) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[3] = QNN[0] (tensor: 1582) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.3_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1650
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1650
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1650) [SAME]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.4_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1670
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 1671
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 1672
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [3] 1655
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1655
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 1670
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 1671
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [3] 1672
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:590   [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[0] expects '1670' but it's at QNN[1]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[1] expects '1671' but it's at QNN[2]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[2] expects '1672' but it's at QNN[3]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[3] expects '1655' but it's at QNN[0]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[0] = QNN[1] (tensor: 1670) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[1] = QNN[2] (tensor: 1671) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[2] = QNN[3] (tensor: 1672) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[3] = QNN[0] (tensor: 1655) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.4_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1723
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1723
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1723) [SAME]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.5_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1743
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 1744
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 1745
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [3] 1728
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1728
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 1743
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 1744
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [3] 1745
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:590   [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[0] expects '1743' but it's at QNN[1]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[1] expects '1744' but it's at QNN[2]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[2] expects '1745' but it's at QNN[3]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[3] expects '1728' but it's at QNN[0]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[0] = QNN[1] (tensor: 1743) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[1] = QNN[2] (tensor: 1744) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[2] = QNN[3] (tensor: 1745) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[3] = QNN[0] (tensor: 1728) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.5_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1796
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1796
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1796) [SAME]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.6_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1816
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 1817
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 1818
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [3] 1801
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1801
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 1816
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 1817
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [3] 1818
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:590   [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[0] expects '1816' but it's at QNN[1]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[1] expects '1817' but it's at QNN[2]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[2] expects '1818' but it's at QNN[3]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[3] expects '1801' but it's at QNN[0]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[0] = QNN[1] (tensor: 1816) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[1] = QNN[2] (tensor: 1817) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[2] = QNN[3] (tensor: 1818) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[3] = QNN[0] (tensor: 1801) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.6_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1869
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1869
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1869) [SAME]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.7_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1889
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 1890
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 1891
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [3] 1874
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1874
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 1889
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 1890
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [3] 1891
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:590   [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[0] expects '1889' but it's at QNN[1]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[1] expects '1890' but it's at QNN[2]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[2] expects '1891' but it's at QNN[3]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[3] expects '1874' but it's at QNN[0]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[0] = QNN[1] (tensor: 1889) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[1] = QNN[2] (tensor: 1890) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[2] = QNN[3] (tensor: 1891) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[3] = QNN[0] (tensor: 1874) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.7_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1942
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1942
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 1942) [SAME]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.8_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 1962
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 1963
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 1964
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [3] 1947
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 1947
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 1962
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 1963
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [3] 1964
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:590   [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[0] expects '1962' but it's at QNN[1]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[1] expects '1963' but it's at QNN[2]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[2] expects '1964' but it's at QNN[3]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[3] expects '1947' but it's at QNN[0]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[0] = QNN[1] (tensor: 1962) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[1] = QNN[2] (tensor: 1963) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[2] = QNN[3] (tensor: 1964) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[3] = QNN[0] (tensor: 1947) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.8_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2015
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2015
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2015) [SAME]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.9_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2035
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2036
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2037
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [3] 2020
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2020
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2035
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2036
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [3] 2037
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:590   [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[0] expects '2035' but it's at QNN[1]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[1] expects '2036' but it's at QNN[2]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[2] expects '2037' but it's at QNN[3]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[3] expects '2020' but it's at QNN[0]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[0] = QNN[1] (tensor: 2035) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[1] = QNN[2] (tensor: 2036) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[2] = QNN[3] (tensor: 2037) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[3] = QNN[0] (tensor: 2020) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.9_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2088
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2088
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2088) [SAME]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.10_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2108
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2109
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2110
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [3] 2093
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2093
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2108
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2109
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [3] 2110
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:590   [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[0] expects '2108' but it's at QNN[1]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[1] expects '2109' but it's at QNN[2]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[2] expects '2110' but it's at QNN[3]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[3] expects '2093' but it's at QNN[0]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[0] = QNN[1] (tensor: 2108) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[1] = QNN[2] (tensor: 2109) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[2] = QNN[3] (tensor: 2110) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[3] = QNN[0] (tensor: 2093) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.10_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2161
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2161
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2161) [SAME]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.11_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2181
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2182
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2183
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [3] 2166
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2166
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2181
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2182
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [3] 2183
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:590   [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[0] expects '2181' but it's at QNN[1]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[1] expects '2182' but it's at QNN[2]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[2] expects '2183' but it's at QNN[3]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[3] expects '2166' but it's at QNN[0]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[0] = QNN[1] (tensor: 2181) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[1] = QNN[2] (tensor: 2182) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[2] = QNN[3] (tensor: 2183) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[3] = QNN[0] (tensor: 2166) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.11_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2234
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2234
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2234) [SAME]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.12_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2254
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2255
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2256
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [3] 2239
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2239
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2254
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2255
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [3] 2256
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:590   [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[0] expects '2254' but it's at QNN[1]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[1] expects '2255' but it's at QNN[2]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[2] expects '2256' but it's at QNN[3]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[3] expects '2239' but it's at QNN[0]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[0] = QNN[1] (tensor: 2254) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[1] = QNN[2] (tensor: 2255) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[2] = QNN[3] (tensor: 2256) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[3] = QNN[0] (tensor: 2239) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.12_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2307
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2307
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2307) [SAME]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.13_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2327
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2328
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2329
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [3] 2312
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2312
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2327
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2328
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [3] 2329
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:590   [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[0] expects '2327' but it's at QNN[1]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[1] expects '2328' but it's at QNN[2]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[2] expects '2329' but it's at QNN[3]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[3] expects '2312' but it's at QNN[0]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[0] = QNN[1] (tensor: 2327) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[1] = QNN[2] (tensor: 2328) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[2] = QNN[3] (tensor: 2329) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[3] = QNN[0] (tensor: 2312) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.13_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2380
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2380
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2380) [SAME]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.14_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2400
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2401
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2402
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [3] 2385
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2385
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2400
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2401
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [3] 2402
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:590   [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[0] expects '2400' but it's at QNN[1]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[1] expects '2401' but it's at QNN[2]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[2] expects '2402' but it's at QNN[3]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[3] expects '2385' but it's at QNN[0]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[0] = QNN[1] (tensor: 2400) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[1] = QNN[2] (tensor: 2401) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[2] = QNN[3] (tensor: 2402) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[3] = QNN[0] (tensor: 2385) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.14_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2453
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2453
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2453) [SAME]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.15_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2473
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2474
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2475
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [3] 2458
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2458
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2473
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2474
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [3] 2475
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:590   [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[0] expects '2473' but it's at QNN[1]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[1] expects '2474' but it's at QNN[2]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[2] expects '2475' but it's at QNN[3]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[3] expects '2458' but it's at QNN[0]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[0] = QNN[1] (tensor: 2473) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[1] = QNN[2] (tensor: 2474) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[2] = QNN[3] (tensor: 2475) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[3] = QNN[0] (tensor: 2458) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.15_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2526
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2526
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2526) [SAME]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.16_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2546
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2547
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2548
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [3] 2531
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2531
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2546
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2547
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [3] 2548
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:590   [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[0] expects '2546' but it's at QNN[1]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[1] expects '2547' but it's at QNN[2]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[2] expects '2548' but it's at QNN[3]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[3] expects '2531' but it's at QNN[0]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[0] = QNN[1] (tensor: 2546) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[1] = QNN[2] (tensor: 2547) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[2] = QNN[3] (tensor: 2548) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[3] = QNN[0] (tensor: 2531) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.16_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2599
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2599
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2599) [SAME]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.17_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2619
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2620
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2621
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [3] 2604
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2604
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2619
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2620
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [3] 2621
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:590   [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[0] expects '2619' but it's at QNN[1]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[1] expects '2620' but it's at QNN[2]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[2] expects '2621' but it's at QNN[3]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[3] expects '2604' but it's at QNN[0]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[0] = QNN[1] (tensor: 2619) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[1] = QNN[2] (tensor: 2620) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[2] = QNN[3] (tensor: 2621) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[3] = QNN[0] (tensor: 2604) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.17_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2672
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2672
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2672) [SAME]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.18_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2692
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2693
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2694
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [3] 2677
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2677
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2692
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2693
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [3] 2694
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:590   [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[0] expects '2692' but it's at QNN[1]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[1] expects '2693' but it's at QNN[2]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[2] expects '2694' but it's at QNN[3]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[3] expects '2677' but it's at QNN[0]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[0] = QNN[1] (tensor: 2692) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[1] = QNN[2] (tensor: 2693) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[2] = QNN[3] (tensor: 2694) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[3] = QNN[0] (tensor: 2677) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.18_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2745
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2745
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2745) [SAME]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.19_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2765
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2766
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2767
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [3] 2750
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2750
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2765
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2766
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [3] 2767
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:590   [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[0] expects '2765' but it's at QNN[1]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[1] expects '2766' but it's at QNN[2]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[2] expects '2767' but it's at QNN[3]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[3] expects '2750' but it's at QNN[0]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[0] = QNN[1] (tensor: 2765) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[1] = QNN[2] (tensor: 2766) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[2] = QNN[3] (tensor: 2767) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[3] = QNN[0] (tensor: 2750) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.19_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2818
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2818
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2818) [SAME]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.20_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2838
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2839
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2840
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [3] 2823
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2823
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2838
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2839
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [3] 2840
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:590   [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[0] expects '2838' but it's at QNN[1]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[1] expects '2839' but it's at QNN[2]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[2] expects '2840' but it's at QNN[3]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[3] expects '2823' but it's at QNN[0]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[0] = QNN[1] (tensor: 2838) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[1] = QNN[2] (tensor: 2839) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[2] = QNN[3] (tensor: 2840) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[3] = QNN[0] (tensor: 2823) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.20_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2891
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2891
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2891) [SAME]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.21_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2911
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2912
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2913
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [3] 2896
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2896
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2911
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2912
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [3] 2913
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:590   [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[0] expects '2911' but it's at QNN[1]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[1] expects '2912' but it's at QNN[2]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[2] expects '2913' but it's at QNN[3]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[3] expects '2896' but it's at QNN[0]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[0] = QNN[1] (tensor: 2911) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[1] = QNN[2] (tensor: 2912) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[2] = QNN[3] (tensor: 2913) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[3] = QNN[0] (tensor: 2896) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.21_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2964
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2964
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 2964) [SAME]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.22_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 2984
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 2985
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 2986
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [3] 2969
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 2969
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 2984
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 2985
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [3] 2986
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:590   [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[0] expects '2984' but it's at QNN[1]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[1] expects '2985' but it's at QNN[2]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[2] expects '2986' but it's at QNN[3]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[3] expects '2969' but it's at QNN[0]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[0] = QNN[1] (tensor: 2984) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[1] = QNN[2] (tensor: 2985) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[2] = QNN[3] (tensor: 2986) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[3] = QNN[0] (tensor: 2969) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.22_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 3037
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 3037
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 3037) [SAME]
[INFO] /root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp:187 query_states_raw shape: [1, 32, 1, 2048]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.23_1'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 3057
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [1] 3058
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [2] 3059
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [3] 3042
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (4 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 3042
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [1] 3057
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [2] 3058
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [3] 3059
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:590   [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[0] expects '3057' but it's at QNN[1]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[1] expects '3058' but it's at QNN[2]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[2] expects '3059' but it's at QNN[3]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:593     Mismatch: MLLM[3] expects '3042' but it's at QNN[0]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[0] = QNN[1] (tensor: 3057) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[1] = QNN[2] (tensor: 3058) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[2] = QNN[3] (tensor: 3059) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:606   Mapping: MLLM[3] = QNN[0] (tensor: 3042) [REORDERED]
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:564 QNNBackend::graphExecute: Checking output order for graph 'model.layers.23_2'
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:565   MLLM Expected Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:567     [0] 3110
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:569   QNN Output Order (1 outputs):
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:572     [0] 3110
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:596   [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed
[INFO] /root/mllm_v2/mllm/backends/qnn/QNNBackend.cpp:608   Mapping: MLLM[0] = QNN[0] (tensor: 3110) [SAME]
token: 2121 As
Error: Received signal11 - SIGSEGV (Segmentation violation)
Stack trace:
#0 0x5d5f57545c
#1 0x5d5f57527c
#2 0x5d5f574f68
#3 0x74c41cf860 __kernel_rt_sigreturn
#4 0x7222ccdff4
#5 0x7222caf750
#6 0x7222c8b748
#7 0x7222e6ea74
#8 0x7222e6e42c
#9 0x7222df9e14
#10 0x7222dfac68
#11 0x7222dfcad4
#12 0x74c23633f0 __cxa_finalize
#13 0x74c236855c exit
#14 0x74c235c158
Possible causes: invalid memory access, dangling pointer, stack overflow.
Shutting down...
```