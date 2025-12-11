// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <tvm/ffi/any.h>
#include <tvm/ffi/string.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/reflection/registry.h>
#include <memory>

#include "mllm/backends/qnn/aot/QnnTargetMachine.hpp"
#include "mllm/ffi/qualcomm/QnnAOT.hh"

#ifdef MLLM_QUALCOMM_QNN_AOT_ON_X86_ENABLE

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  refl::ObjectDef<::mllm::ffi::QcomHTPArchObj>();

  refl::GlobalDef().def("mllm.qualcomm.QcomHTPArch.NONE", []() {
    auto ret = mllm::qnn::aot::QcomHTPArch::NONE;
    return mllm::ffi::QcomHTPArch(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomHTPArch.V68", []() {
    auto ret = mllm::qnn::aot::QcomHTPArch::V68;
    return mllm::ffi::QcomHTPArch(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomHTPArch.V69", []() {
    auto ret = mllm::qnn::aot::QcomHTPArch::V69;
    return mllm::ffi::QcomHTPArch(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomHTPArch.V73", []() {
    auto ret = mllm::qnn::aot::QcomHTPArch::V73;
    return mllm::ffi::QcomHTPArch(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomHTPArch.V75", []() {
    auto ret = mllm::qnn::aot::QcomHTPArch::V79;
    return mllm::ffi::QcomHTPArch(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomHTPArch.V79", []() {
    auto ret = mllm::qnn::aot::QcomHTPArch::V79;
    return mllm::ffi::QcomHTPArch(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomHTPArch.V81", []() {
    auto ret = mllm::qnn::aot::QcomHTPArch::V81;
    return mllm::ffi::QcomHTPArch(ret);
  });

  refl::ObjectDef<::mllm::ffi::QcomChipsetObj>();

  refl::GlobalDef().def("mllm.qualcomm.QcomChipset.UNKNOWN_SM", []() {
    auto ret = mllm::qnn::aot::QcomChipset::UNKNOWN_SM;
    return mllm::ffi::QcomChipset(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomChipset.SA8295", []() {
    auto ret = mllm::qnn::aot::QcomChipset::SA8295;
    return mllm::ffi::QcomChipset(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomChipset.SM8350", []() {
    auto ret = mllm::qnn::aot::QcomChipset::SM8350;
    return mllm::ffi::QcomChipset(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomChipset.SM8450", []() {
    auto ret = mllm::qnn::aot::QcomChipset::SM8450;
    return mllm::ffi::QcomChipset(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomChipset.SM8475", []() {
    auto ret = mllm::qnn::aot::QcomChipset::SM8475;
    return mllm::ffi::QcomChipset(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomChipset.SM8550", []() {
    auto ret = mllm::qnn::aot::QcomChipset::SM8550;
    return mllm::ffi::QcomChipset(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomChipset.SM8650", []() {
    auto ret = mllm::qnn::aot::QcomChipset::SM8650;
    return mllm::ffi::QcomChipset(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomChipset.SM8750", []() {
    auto ret = mllm::qnn::aot::QcomChipset::SM8750;
    return mllm::ffi::QcomChipset(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomChipset.SM8850", []() {
    auto ret = mllm::qnn::aot::QcomChipset::SM8850;
    return mllm::ffi::QcomChipset(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomChipset.SSG2115P", []() {
    auto ret = mllm::qnn::aot::QcomChipset::SSG2115P;
    return mllm::ffi::QcomChipset(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomChipset.SSG2125P", []() {
    auto ret = mllm::qnn::aot::QcomChipset::SSG2125P;
    return mllm::ffi::QcomChipset(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomChipset.SXR1230P", []() {
    auto ret = mllm::qnn::aot::QcomChipset::SXR1230P;
    return mllm::ffi::QcomChipset(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomChipset.SXR2230P", []() {
    auto ret = mllm::qnn::aot::QcomChipset::SXR2230P;
    return mllm::ffi::QcomChipset(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomChipset.SXR2330P", []() {
    auto ret = mllm::qnn::aot::QcomChipset::SXR2330P;
    return mllm::ffi::QcomChipset(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomChipset.QCS9100", []() {
    auto ret = mllm::qnn::aot::QcomChipset::QCS9100;
    return mllm::ffi::QcomChipset(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomChipset.SAR2230P", []() {
    auto ret = mllm::qnn::aot::QcomChipset::SAR2230P;
    return mllm::ffi::QcomChipset(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomChipset.SA8255", []() {
    auto ret = mllm::qnn::aot::QcomChipset::SA8255;
    return mllm::ffi::QcomChipset(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomChipset.SW6100", []() {
    auto ret = mllm::qnn::aot::QcomChipset::SW6100;
    return mllm::ffi::QcomChipset(ret);
  });

  refl::ObjectDef<::mllm::ffi::QcomTryBestPerformanceObj>();

  refl::GlobalDef().def("mllm.qualcomm.QcomTryBestPerformance.HtpDefault", []() {
    auto ret = mllm::qnn::aot::QcomTryBestPerformance::kHtpDefault;
    return mllm::ffi::QcomTryBestPerformance(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomTryBestPerformance.HtpSustainedHighPerformance", []() {
    auto ret = mllm::qnn::aot::QcomTryBestPerformance::kHtpSustainedHighPerformance;
    return mllm::ffi::QcomTryBestPerformance(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomTryBestPerformance.HtpBurst", []() {
    auto ret = mllm::qnn::aot::QcomTryBestPerformance::kHtpBurst;
    return mllm::ffi::QcomTryBestPerformance(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomTryBestPerformance.HtpHighPerformance", []() {
    auto ret = mllm::qnn::aot::QcomTryBestPerformance::kHtpHighPerformance;
    return mllm::ffi::QcomTryBestPerformance(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomTryBestPerformance.HtpPowerSaver", []() {
    auto ret = mllm::qnn::aot::QcomTryBestPerformance::kHtpPowerSaver;
    return mllm::ffi::QcomTryBestPerformance(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomTryBestPerformance.HtpLowPowerSaver", []() {
    auto ret = mllm::qnn::aot::QcomTryBestPerformance::kHtpLowPowerSaver;
    return mllm::ffi::QcomTryBestPerformance(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomTryBestPerformance.HtpHighPowerSaver", []() {
    auto ret = mllm::qnn::aot::QcomTryBestPerformance::kHtpHighPowerSaver;
    return mllm::ffi::QcomTryBestPerformance(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomTryBestPerformance.HtpLowBalanced", []() {
    auto ret = mllm::qnn::aot::QcomTryBestPerformance::kHtpLowBalanced;
    return mllm::ffi::QcomTryBestPerformance(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomTryBestPerformance.HtpBalanced", []() {
    auto ret = mllm::qnn::aot::QcomTryBestPerformance::kHtpBalanced;
    return mllm::ffi::QcomTryBestPerformance(ret);
  });

  refl::ObjectDef<::mllm::ffi::QcomSecurityPDSessionObj>();

  refl::GlobalDef().def("mllm.qualcomm.QcomSecurityPDSession.HtpUnsignedPd", []() {
    auto ret = mllm::qnn::aot::QcomSecurityPDSession::kHtpUnsignedPd;
    return mllm::ffi::QcomSecurityPDSession(ret);
  });
  refl::GlobalDef().def("mllm.qualcomm.QcomSecurityPDSession.HtpSignedPd", []() {
    auto ret = mllm::qnn::aot::QcomSecurityPDSession::kHtpSignedPd;
    return mllm::ffi::QcomSecurityPDSession(ret);
  });

  refl::ObjectDef<mllm::ffi::QcomTargetMachineObj>().def_static(
      "__create__", [](const mllm::ffi::QcomChipset& chipset, const mllm::ffi::QcomHTPArch& arch,
                       const mllm::ffi::QcomTryBestPerformance& perf, const mllm::ffi::QcomSecurityPDSession& pd_session) {
        auto tm = mllm::qnn::aot::QcomTargetMachine{
            .soc_htp_chipset = chipset.get()->chipset_,
            .soc_htp_arch = arch.get()->htp_arch_,
            .soc_htp_performance = perf.get()->perf_,
            .soc_htp_security_pd_session = pd_session.get()->pd_,
        };
        return ::mllm::ffi::QcomTargetMachine(tm);
      });

  refl::ObjectDef<mllm::ffi::QnnAOTEnvObj>().def_static(
      "__create__", [](const mllm::ffi::QcomTargetMachine& machine, const std::string& path) -> mllm::ffi::QnnAOTEnv {
        if (path.empty()) {
          auto tm = machine.get()->target_machine_;
          auto s = std::make_shared<::mllm::qnn::aot::QnnAOTEnv>(tm);
          return ::mllm::ffi::QnnAOTEnv(s);
        } else {
          auto tm = machine.get()->target_machine_;
          auto s = std::make_shared<::mllm::qnn::aot::QnnAOTEnv>(path, tm);
          return ::mllm::ffi::QnnAOTEnv(s);
        }
      });

  refl::ObjectDef<::mllm::ffi::QnnDeviceAndContextObj>();

  refl::GlobalDef().def("mllm.qualcomm.QnnAOTEnv.createContext", [](const mllm::ffi::QnnAOTEnv& self, const std::string& name) {
    auto s = self.get()->qnn_aot_env_ptr_->createContext(name);
    return mllm::ffi::QnnDeviceAndContext(s);
  });
}

#endif
