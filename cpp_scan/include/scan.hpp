#ifndef SCAN_H__
#define SCAN_H__

#include <stddef.h>
#include <stdint.h>

#define MAX_BLOCK_SZ 1024
#define LOG_NUM_BANKS 5

void scan(int32_t* d_out, int32_t const* d_in, size_t len);

#endif // SCAN_H__
