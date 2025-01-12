#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

GGML_API ggml_backend_t ggml_backend_cartesi_init(void);
GGML_API bool ggml_backend_is_cartesi(ggml_backend_t backend);
GGML_API ggml_backend_reg_t ggml_backend_cartesi_reg(void);

#ifdef  __cplusplus
}
#endif 