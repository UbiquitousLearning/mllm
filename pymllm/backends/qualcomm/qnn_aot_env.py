from pymllm.ffi import is_qnn_aot_on_x86_enabled

if is_qnn_aot_on_x86_enabled():
    from pymllm.ffi import (
        QnnDeviceAndContext,
        QnnAOTEnv,
        QcomChipset,
        QcomHTPArch,
        QcomSecurityPDSession,
        QcomTargetMachine,
        QcomTryBestPerformance,
    )
else:
    # Define placeholder classes when QNN AOT is not enabled
    QnnDeviceAndContext = None
    QnnAOTEnv = None
    QcomChipset = None
    QcomHTPArch = None
    QcomSecurityPDSession = None
    QcomTargetMachine = None
    QcomTryBestPerformance = None