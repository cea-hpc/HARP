#ifndef SCAN_H__
#define SCAN_H__

#include <cstdint>

auto scan_inner(
	int32_t* d_out,
	int32_t const* d_in,
	size_t len
) -> void;

#endif // SCAN_H__
