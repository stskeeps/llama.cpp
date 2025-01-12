#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <memory>
#include <vector>
#include <string>

// Cartesi API types and functions
typedef struct cartesi_buffer* cartesi_buffer_t;
typedef struct cartesi_context* cartesi_context_t;

extern "C" {
    cartesi_context_t cartesi_create_context(void);
    void cartesi_destroy_context(cartesi_context_t ctx);
    cartesi_buffer_t cartesi_alloc_buffer(cartesi_context_t ctx, size_t size);
    void cartesi_free_buffer(cartesi_buffer_t buffer);
    void cartesi_memset_buffer(cartesi_buffer_t buffer, int value, size_t size);
    void cartesi_memcpy_to_buffer(cartesi_buffer_t dst, const void* src, size_t size);
    void cartesi_memcpy_from_buffer(void* dst, cartesi_buffer_t src, size_t size);
    
    // Matrix multiplication function using Cartesi's software floating point
    void cartesi_matmul(cartesi_context_t ctx,
                       cartesi_buffer_t dst,
                       cartesi_buffer_t src0,
                       cartesi_buffer_t src1,
                       int ne0,  // dst cols (n)
                       int ne1,  // dst rows (m) 
                       int ne2,  // src0 cols (k)
                       size_t nb0, // dst col stride
                       size_t nb1, // dst row stride
                       size_t nb2, // src0 col stride
                       size_t nb3); // src1 col stride
}

// Backend context
struct ggml_backend_cartesi_context {
    cartesi_context_t ctx;
    size_t max_batch_size;
    bool initialized;

    ggml_backend_cartesi_context() : ctx(nullptr), max_batch_size(1024*1024), initialized(false) {}
};

// Buffer context
struct ggml_backend_cartesi_buffer_context {
    cartesi_buffer_t buffer;
    std::string name;

    explicit ggml_backend_cartesi_buffer_context(cartesi_buffer_t buf) : buffer(buf), name("Cartesi") {}
    ~ggml_backend_cartesi_buffer_context() {
        if (buffer) {
            cartesi_free_buffer(buffer);
        }
    }
};

static void* const cartesi_ptr_base = (void*)(uintptr_t)0x1000;

// Backend implementation
static const char* ggml_backend_cartesi_name(ggml_backend_t backend) {
    return "Cartesi";
    GGML_UNUSED(backend);
}

static void ggml_backend_cartesi_free(ggml_backend_t backend) {
    ggml_backend_cartesi_context* ctx = (ggml_backend_cartesi_context*)backend->context;
    if (ctx->ctx) {
        cartesi_destroy_context(ctx->ctx);
    }
    delete ctx;
    delete backend;
}

static ggml_backend_cartesi_context* ggml_cartesi_init(ggml_backend_dev_t dev) {
    static bool initialized = false;
    static ggml_backend_cartesi_context* backend_ctx = nullptr;

    if (initialized) {
        return backend_ctx;
    }

    initialized = true;
    backend_ctx = new ggml_backend_cartesi_context();
    backend_ctx->ctx = cartesi_create_context();
    backend_ctx->initialized = true;

    return backend_ctx;
}

// Buffer implementation
static void ggml_backend_cartesi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cartesi_buffer_context* ctx = (ggml_backend_cartesi_buffer_context*)buffer->context;
    delete ctx;
}

static void* ggml_backend_cartesi_buffer_get_base(ggml_backend_buffer_t buffer) {
    return cartesi_ptr_base;
    GGML_UNUSED(buffer);
}

static void ggml_backend_cartesi_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor* tensor) {
    // Initialize tensor with cartesi buffer
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
}

static void ggml_backend_cartesi_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor* tensor, const void* data, size_t offset, size_t size) {
    ggml_backend_cartesi_buffer_context* ctx = (ggml_backend_cartesi_buffer_context*)buffer->context;
    cartesi_memcpy_to_buffer(ctx->buffer, data, size);
    GGML_UNUSED(tensor);
    GGML_UNUSED(offset);
}

static void ggml_backend_cartesi_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor* tensor, void* data, size_t offset, size_t size) {
    ggml_backend_cartesi_buffer_context* ctx = (ggml_backend_cartesi_buffer_context*)buffer->context;
    cartesi_memcpy_from_buffer(data, ctx->buffer, size);
    GGML_UNUSED(tensor);
    GGML_UNUSED(offset);
}

static void ggml_backend_cartesi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cartesi_buffer_context* ctx = (ggml_backend_cartesi_buffer_context*)buffer->context;
    cartesi_memset_buffer(ctx->buffer, value, buffer->size);
}

static ggml_backend_buffer_i ggml_backend_cartesi_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cartesi_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cartesi_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cartesi_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_cartesi_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cartesi_buffer_get_tensor,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_cartesi_buffer_clear,
    /* .reset           = */ nullptr,
};

// Buffer type implementation
static const char* ggml_backend_cartesi_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return "Cartesi";
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cartesi_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cartesi_context* ctx = ggml_cartesi_init(buft->device);
    
    cartesi_buffer_t buffer = cartesi_alloc_buffer(ctx->ctx, size);
    if (!buffer) {
        return nullptr;
    }

    ggml_backend_cartesi_buffer_context* buffer_ctx = new ggml_backend_cartesi_buffer_context(buffer);
    return ggml_backend_buffer_init(buft, ggml_backend_cartesi_buffer_interface, buffer_ctx, size);
}

static ggml_backend_buffer_type_i ggml_backend_cartesi_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cartesi_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_cartesi_buffer_type_alloc_buffer,
    /* .get_alignment    = */ nullptr,
    /* .get_max_size     = */ nullptr,
    /* .get_alloc_size   = */ nullptr,
    /* .supports_backend = */ nullptr,
    /* .is_host         = */ nullptr,
};

// Compute implementation
static void ggml_backend_cartesi_mul_mat(ggml_backend_cartesi_context* ctx, ggml_tensor* dst) {
    const struct ggml_tensor* src0 = dst->src[0];
    const struct ggml_tensor* src1 = dst->src[1];

    const int ne0 = dst->ne[0];    // dst cols (n)
    const int ne1 = dst->ne[1];    // dst rows (m)
    const int ne2 = src0->ne[0];   // src0 cols (k)

    const size_t nb0 = dst->nb[0];     // dst col stride
    const size_t nb1 = dst->nb[1];     // dst row stride
    const size_t nb2 = src0->nb[0];    // src0 col stride
    const size_t nb3 = src1->nb[0];    // src1 col stride

    ggml_backend_cartesi_buffer_context* dst_buf = (ggml_backend_cartesi_buffer_context*)dst->buffer->context;
    ggml_backend_cartesi_buffer_context* src0_buf = (ggml_backend_cartesi_buffer_context*)src0->buffer->context;
    ggml_backend_cartesi_buffer_context* src1_buf = (ggml_backend_cartesi_buffer_context*)src1->buffer->context;

    // Call the Cartesi API's matrix multiplication
    cartesi_matmul(ctx->ctx, dst_buf->buffer, src0_buf->buffer, src1_buf->buffer,
                   ne0, ne1, ne2, nb0, nb1, nb2, nb3);
}

static ggml_status ggml_backend_cartesi_graph_compute(ggml_backend_t backend, ggml_cgraph* cgraph) {
    ggml_backend_cartesi_context* ctx = (ggml_backend_cartesi_context*)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor* node = cgraph->nodes[i];

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_cartesi_mul_mat(ctx, node);
                break;
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                // These ops don't require computation
                break;
            default:
                GGML_ASSERT(false && "Operation not supported by Cartesi backend");
                return GGML_STATUS_FAILED;
        }
    }

    return GGML_STATUS_SUCCESS;
}

// Device implementation
static const char* ggml_backend_cartesi_device_name(ggml_backend_dev_t dev) {
    return "Cartesi";
    GGML_UNUSED(dev);
}

static void ggml_backend_cartesi_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props* props) {
    props->name = ggml_backend_cartesi_device_name(dev);
    props->description = "Cartesi Software Floating Point Device";
    props->type = GGML_BACKEND_DEVICE_TYPE_CPU;  // Changed to CPU since it's software floating point
    props->memory_free = 0;
    props->memory_total = 0;
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
    };
}

static bool ggml_backend_cartesi_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor* op) {
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return true;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
    GGML_UNUSED(dev);
}

static ggml_backend_t ggml_backend_cartesi_device_init(ggml_backend_dev_t dev, const char* params) {
    ggml_backend_cartesi_context* ctx = ggml_cartesi_init(dev);

    static ggml_backend_i backend_i = {
        /* .get_name                = */ ggml_backend_cartesi_name,
        /* .free                    = */ ggml_backend_cartesi_free,
        /* .set_tensor_async        = */ nullptr,
        /* .get_tensor_async        = */ nullptr,
        /* .cpy_tensor_async        = */ nullptr,
        /* .synchronize             = */ nullptr,
        /* .graph_plan_create       = */ nullptr,
        /* .graph_plan_free         = */ nullptr,
        /* .graph_plan_update       = */ nullptr,
        /* .graph_plan_compute      = */ nullptr,
        /* .graph_compute           = */ ggml_backend_cartesi_graph_compute,
        /* .event_record            = */ nullptr,
        /* .event_wait              = */ nullptr,
    };

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_cartesi_guid(),
        /* .interface = */ backend_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    return backend;
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_cartesi_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cartesi_buffer_type();
    GGML_UNUSED(dev);
}

static ggml_backend_device_i ggml_backend_cartesi_device_interface = {
    /* .get_name             = */ ggml_backend_cartesi_device_name,
    /* .get_description      = */ nullptr,
    /* .get_memory          = */ nullptr,
    /* .get_type            = */ nullptr,
    /* .get_props           = */ ggml_backend_cartesi_device_get_props,
    /* .init_backend        = */ ggml_backend_cartesi_device_init,
    /* .get_buffer_type     = */ ggml_backend_cartesi_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op         = */ ggml_backend_cartesi_device_supports_op,
    /* .supports_buft       = */ nullptr,
    /* .offload_op          = */ nullptr,
    /* .event_new           = */ nullptr,
    /* .event_free          = */ nullptr,
    /* .event_synchronize   = */ nullptr,
};

// Backend registry implementation
static const char* ggml_backend_cartesi_reg_name(ggml_backend_reg_t reg) {
    return "Cartesi";
    GGML_UNUSED(reg);
}

static ggml_backend_t ggml_backend_cartesi_reg_init(ggml_backend_reg_t reg, const char* params) {
    return ggml_backend_cartesi_device_init(ggml_backend_reg_dev_get(reg, 0), params);
}

static size_t ggml_backend_cartesi_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_cartesi_reg_get_device(ggml_backend_reg_t reg, size_t idx) {
    GGML_ASSERT(idx == 0);
    static ggml_backend_device device = {
        /* .iface   = */ ggml_backend_cartesi_device_interface,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
    return &device;
}

static ggml_backend_reg_i ggml_backend_cartesi_reg_interface = {
    /* .get_name         = */ ggml_backend_cartesi_reg_name,
    /* .get_device_count = */ ggml_backend_cartesi_reg_get_device_count,
    /* .get_device       = */ ggml_backend_cartesi_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

// Public API
GGML_API ggml_backend_t ggml_backend_cartesi_init(void) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_cartesi_reg(), 0);
    return ggml_backend_cartesi_device_init(dev, nullptr);
}

GGML_API bool ggml_backend_is_cartesi(ggml_backend_t backend) {
    return backend && backend->iface.get_name == ggml_backend_cartesi_name;
}

GGML_API ggml_backend_reg_t ggml_backend_cartesi_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cartesi_reg_interface,
        /* .context     = */ nullptr,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cartesi_reg) 