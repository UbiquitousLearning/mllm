import pymllm as mllm
from pymllm.backends.qualcomm.qnn_aot_env import QnnAOTEnv, QnnDeviceAndContext

qnn_aot_env: QnnAOTEnv = QnnAOTEnv()

if __name__ == "__main__":
    mllm.echo("Testing mllm's tvm-ffi abi compatibility")
    qnn_context: QnnDeviceAndContext = qnn_aot_env.create_context("model.layer.0")
