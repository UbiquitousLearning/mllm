import pymllm as mllm
from pymllm.backends.qualcomm.qnn_aot_env import (
    QnnAOTEnv,
    QnnDeviceAndContext,
    QcomTryBestPerformance,
    QcomSecurityPDSession,
    QcomTargetMachine,
    QcomChipset,
    QcomHTPArch,
)


qnn_aot_env: QnnAOTEnv = QnnAOTEnv(
    machine=QcomTargetMachine(
        soc_htp_chipset=QcomChipset.SM8850(),
        soc_htp_arch=QcomHTPArch.V81(),
        soc_htp_performance=QcomTryBestPerformance.HtpBurst(),
        soc_htp_security_pd_session=QcomSecurityPDSession.HtpUnsignedPd(),
        soc_htp_vtcm=8,  # in MB
    ),
    path="/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/",
)

if __name__ == "__main__":
    mllm.echo("Testing tvm-ffi compatibility")
    qnn_context: QnnDeviceAndContext = qnn_aot_env.create_context(
        "context.0", weights_sharing=False
    )
