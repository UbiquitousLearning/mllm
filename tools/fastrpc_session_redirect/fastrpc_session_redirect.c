#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <remote.h>
#include <dspqueue.h>

enum {
    BASE_CDSP_DOMAIN = 3,
};

static pthread_mutex_t redirect_mutex = PTHREAD_MUTEX_INITIALIZER;
static int effective_domain = -1;
static uint32_t alternate_session_id = 0;
static void *cdsprpc_library = NULL;

typedef int (*session_control_fn)(uint32_t, void *, uint32_t);
typedef int (*handle64_open_fn)(const char *, remote_handle64 *);
typedef int (*handle_open_fn)(const char *, remote_handle *);
typedef int (*fastrpc_mmap_fn)(int, int, void *, int, size_t,
                              enum fastrpc_map_flags);
typedef int (*fastrpc_munmap_fn)(int, int, void *, size_t);
typedef AEEResult (*dspqueue_create_fn)(
    int, uint32_t, uint32_t, uint32_t, dspqueue_callback_t,
    dspqueue_callback_t, void *, dspqueue_t *);

static void *next_symbol(const char *name) {
    void *symbol = dlsym(RTLD_NEXT, name);
    if (!symbol) {
        (void)dlerror();
        if (!cdsprpc_library) {
            cdsprpc_library =
                dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
        }
        if (cdsprpc_library) {
            symbol = dlsym(cdsprpc_library, name);
        }
    }
    if (!symbol) {
        fprintf(stderr, "FastRPC redirect: dlsym(%s) failed: %s\n", name,
                dlerror());
    }
    return symbol;
}

static int ensure_alternate_session(session_control_fn real_control) {
    pthread_mutex_lock(&redirect_mutex);
    if (effective_domain >= 0) {
        pthread_mutex_unlock(&redirect_mutex);
        return 0;
    }

    char domain_name[] = "cdsp";
    char session_name[] = "mllm_qnn_recovery";
    remote_rpc_reserve_new_session_t reserve = {
        .domain_name = domain_name,
        .domain_name_len = (uint32_t)strlen(domain_name),
        .session_name = session_name,
        .session_name_len = (uint32_t)strlen(session_name),
    };
    int status = real_control(FASTRPC_RESERVE_NEW_SESSION, &reserve,
                              sizeof(reserve));
    if (status == 0) {
        effective_domain = (int)reserve.effective_domain_id;
        alternate_session_id = reserve.session_id;
        fprintf(stderr,
                "FastRPC redirect: CDSP domain %d/session 0 -> domain %d/"
                "session %u\n",
                BASE_CDSP_DOMAIN, effective_domain, alternate_session_id);
    } else {
        fprintf(stderr,
                "FastRPC redirect: reserve alternate session failed: %d "
                "(0x%x)\n",
                status, (unsigned int)status);
    }
    pthread_mutex_unlock(&redirect_mutex);
    return status;
}

static int request_has_domain(uint32_t request) {
    switch (request) {
        case FASTRPC_THREAD_PARAMS:
        case DSPRPC_CONTROL_UNSIGNED_MODULE:
        case FASTRPC_RELATIVE_THREAD_PRIORITY:
        case FASTRPC_REMOTE_PROCESS_KILL:
        case FASTRPC_SESSION_CLOSE:
        case FASTRPC_CONTROL_PD_DUMP:
        case FASTRPC_REMOTE_PROCESS_EXCEPTION:
        case FASTRPC_REMOTE_PROCESS_TYPE:
        case FASTRPC_PD_INITMEM_SIZE:
            return 1;
        default:
            return 0;
    }
}

int remote_session_control(uint32_t request, void *data, uint32_t length) {
    session_control_fn real_control =
        (session_control_fn)next_symbol("remote_session_control");
    if (!real_control) {
        return -1;
    }
    if (request_has_domain(request) && data && length >= sizeof(int)
        && *(int *)data == BASE_CDSP_DOMAIN) {
        int status = ensure_alternate_session(real_control);
        if (status != 0) {
            return status;
        }
        int original_domain = *(int *)data;
        *(int *)data = effective_domain;
        status = real_control(request, data, length);
        *(int *)data = original_domain;
        return status;
    }
    if (request == FASTRPC_REGISTER_STATUS_NOTIFICATIONS && data
        && length >= sizeof(remote_rpc_notif_register_t)) {
        remote_rpc_notif_register_t *registration =
            (remote_rpc_notif_register_t *)data;
        if (registration->domain == BASE_CDSP_DOMAIN) {
            int status = ensure_alternate_session(real_control);
            if (status != 0) {
                return status;
            }
            int original_domain = registration->domain;
            registration->domain = effective_domain;
            status = real_control(request, data, length);
            registration->domain = original_domain;
            return status;
        }
    }
    return real_control(request, data, length);
}

static char *redirect_uri(const char *name) {
    if (!name || strstr(name, "_dom=cdsp") == NULL) {
        return NULL;
    }
    const char *session = strstr(name, "_session=");
    if (session) {
        const char *value = session + strlen("_session=");
        if (*value != '0' || (value[1] >= '0' && value[1] <= '9')) {
            return NULL;
        }
        char *redirected = strdup(name);
        if (redirected) {
            redirected[value - name] = (char)('0' + alternate_session_id);
        }
        return redirected;
    }
    const size_t bytes = strlen(name) + 32u;
    char *redirected = (char *)malloc(bytes);
    if (!redirected) {
        return NULL;
    }
    (void)snprintf(redirected, bytes, "%s&_session=%u", name,
                   alternate_session_id);
    return redirected;
}

int remote_handle64_open(const char *name, remote_handle64 *handle) {
    handle64_open_fn real_open =
        (handle64_open_fn)next_symbol("remote_handle64_open");
    session_control_fn real_control =
        (session_control_fn)next_symbol("remote_session_control");
    if (!real_open || !real_control
        || ensure_alternate_session(real_control) != 0) {
        return -1;
    }
    char *redirected = redirect_uri(name);
    int status = real_open(redirected ? redirected : name, handle);
    if (status != 0) {
        fprintf(stderr, "FastRPC redirect: remote_handle64_open failed: "
                        "%d (0x%x), uri=%s\n",
                status, (unsigned int)status,
                redirected ? redirected : name);
    }
    free(redirected);
    return status;
}

int remote_handle_open(const char *name, remote_handle *handle) {
    handle_open_fn real_open =
        (handle_open_fn)next_symbol("remote_handle_open");
    session_control_fn real_control =
        (session_control_fn)next_symbol("remote_session_control");
    if (!real_open || !real_control
        || ensure_alternate_session(real_control) != 0) {
        return -1;
    }
    char *redirected = redirect_uri(name);
    int status = real_open(redirected ? redirected : name, handle);
    free(redirected);
    return status;
}

int fastrpc_mmap(int domain, int fd, void *address, int offset, size_t length,
                 enum fastrpc_map_flags flags) {
    fastrpc_mmap_fn real_mmap =
        (fastrpc_mmap_fn)next_symbol("fastrpc_mmap");
    session_control_fn real_control =
        (session_control_fn)next_symbol("remote_session_control");
    if (!real_mmap || !real_control) {
        return -1;
    }
    if (domain == BASE_CDSP_DOMAIN) {
        int status = ensure_alternate_session(real_control);
        if (status != 0) {
            return status;
        }
        domain = effective_domain;
    }
    return real_mmap(domain, fd, address, offset, length, flags);
}

int fastrpc_munmap(int domain, int fd, void *address, size_t length) {
    fastrpc_munmap_fn real_munmap =
        (fastrpc_munmap_fn)next_symbol("fastrpc_munmap");
    if (!real_munmap) {
        return -1;
    }
    if (domain == BASE_CDSP_DOMAIN && effective_domain >= 0) {
        domain = effective_domain;
    }
    return real_munmap(domain, fd, address, length);
}

AEEResult dspqueue_create(int domain, uint32_t flags,
                          uint32_t request_queue_size,
                          uint32_t response_queue_size,
                          dspqueue_callback_t packet_callback,
                          dspqueue_callback_t error_callback,
                          void *callback_context, dspqueue_t *queue) {
    dspqueue_create_fn real_create =
        (dspqueue_create_fn)next_symbol("dspqueue_create");
    session_control_fn real_control =
        (session_control_fn)next_symbol("remote_session_control");
    if (!real_create || !real_control) {
        return -1;
    }
    if (domain == BASE_CDSP_DOMAIN) {
        int status = ensure_alternate_session(real_control);
        if (status != 0) {
            return status;
        }
        domain = effective_domain;
    }
    return real_create(domain, flags, request_queue_size, response_queue_size,
                       packet_callback, error_callback, callback_context,
                       queue);
}
