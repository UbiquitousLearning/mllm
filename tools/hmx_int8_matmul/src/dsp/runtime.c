#include "dsp/runtime.h"

#include <HAP_compute_res.h>
#include <HAP_farf.h>
#include <HAP_power.h>
#include <string.h>

static int s_power_context;

static void set_hmx_power(int enabled) {
    HAP_power_request_t request;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_HMX;
    request.hmx.power_up = enabled ? TRUE : FALSE;
    if (HAP_power_set(&s_power_context, &request) != 0) {
        FARF(ALWAYS, "hmx_int8: HMX power request failed (enabled=%d)", enabled);
    }
}

static void set_performance_mode(void) {
    HAP_power_request_t request;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_DCVS_v3;
    request.dcvs_v3.dcvs_enable = TRUE;
    request.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
    request.dcvs_v3.set_latency = TRUE;
    request.dcvs_v3.latency = 100;
    request.dcvs_v3.set_core_params = TRUE;
    request.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_NOM;
    request.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_TURBO_L3;
    request.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_TURBO_L3;
    request.dcvs_v3.set_bus_params = TRUE;
    request.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_NOM;
    request.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_TURBO_L3;
    request.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_TURBO_L3;
    (void)HAP_power_set(&s_power_context, &request);
}

int hmx_runtime_setup(void) {
    /* Direct-path contexts do not pin scarce HMX/VTCM resources. */
    return 0;
}

void hmx_runtime_reset(void) {
    set_hmx_power(0);
}

int hmx_runtime_begin(hmx_runtime_session *session, uint32_t minimum_vtcm_size) {
    if (!session || minimum_vtcm_size == 0) {
        return -1;
    }
    memset(session, 0, sizeof(*session));
    minimum_vtcm_size = (minimum_vtcm_size + 4095u) & ~4095u;
    set_performance_mode();
    set_hmx_power(1);

    compute_res_attr_t request;
    HAP_compute_res_attr_init(&request);
    if (HAP_compute_res_attr_set_vtcm_param(
            &request, minimum_vtcm_size, 1u) != 0 ||
        HAP_compute_res_attr_set_hmx_param(&request, 1) != 0) {
        set_hmx_power(0);
        return -1;
    }
    session->resource_context = HAP_compute_res_acquire(&request, 10000u);
    if (!session->resource_context) {
        FARF(ALWAYS, "hmx_int8: resource acquire failed, VTCM=%u",
             minimum_vtcm_size);
        set_hmx_power(0);
        return -1;
    }
    session->vtcm_base =
        (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&request);
    session->vtcm_size = minimum_vtcm_size;
    if (!session->vtcm_base ||
        HAP_compute_res_hmx_lock2(
            session->resource_context, HAP_COMPUTE_RES_HMX_SHARED) != 0) {
        hmx_runtime_end(session);
        return -1;
    }
    session->hmx_locked = 1;
    return 0;
}

void hmx_runtime_end(hmx_runtime_session *session) {
    if (!session) {
        return;
    }
    if (session->hmx_locked && session->resource_context) {
        (void)HAP_compute_res_hmx_unlock2(
            session->resource_context, HAP_COMPUTE_RES_HMX_SHARED);
    }
    if (session->resource_context) {
        (void)HAP_compute_res_release(session->resource_context);
    }
    memset(session, 0, sizeof(*session));
    set_hmx_power(0);
}
