#include "scan.h"
#include "timer.h"
#include "utils.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>

auto host_scan(
	int32_t* h_out,
	int32_t const* h_in,
	size_t len
) -> void {
	int32_t acc = 0;
	for (size_t i = 0; i < len; ++i) {
		h_out[i] = acc;
		acc += h_in[i];
	}
}

auto main(int32_t argc, char* argv[]) -> int32_t {
	size_t len = 1.5 * (2 << 27);
	if (argc == 2) {
		len = static_cast<size_t>(atoi(argv[1]));
	}

	std::cout
		<< "\n\x1b[1mvector size:\x1b[0m\t" << len
		<< " (" << static_cast<double>(len * sizeof(int32_t)) / std::pow(2.0, 30) << " GiB)"
		<< std::endl;

	// Generate input (default is all ones)
	int32_t* h_in = new int32_t[len];
	for (size_t i = 0; i < len; ++i) {
		h_in[i] = 1;
	}

	// Set up host-side memory for output
	int32_t* h_out_naive = new int32_t[len];
	int32_t* h_out_optim = new int32_t[len];

	// Set up device-side memory for input
	int32_t* d_in;
	checkCudaErrors(cudaMalloc(&d_in, len * sizeof(int32_t)));
	checkCudaErrors(cudaMemcpy(d_in, h_in, len * sizeof(int32_t), cudaMemcpyHostToDevice));

	// Set up device-side memory for output
	int32_t* d_out_optim;
	checkCudaErrors(cudaMalloc(&d_out_optim, len * sizeof(int32_t)));

	// Do CPU scan for reference
	auto h_start = std::clock();
	host_scan(h_out_naive, h_in, len);
	double h_duration =
		static_cast<double>(std::clock() - h_start) / static_cast<double>(CLOCKS_PER_SEC);

	// Do GPU scan
	auto d_start = std::clock();
	scan_inner(d_out_optim, d_in, len);
	double d_duration =
		static_cast<double>(std::clock() - d_start) / static_cast<double>(CLOCKS_PER_SEC);

	// Copy device output array to host output array
	// And free device-side memory
	checkCudaErrors(cudaMemcpy(
		h_out_optim,
		d_out_optim,
		len * sizeof(int32_t),
		cudaMemcpyDeviceToHost
	));
	checkCudaErrors(cudaFree(d_out_optim));
	checkCudaErrors(cudaFree(d_in));

	// Check for any mismatches between outputs of CPU and GPU
	ptrdiff_t diff = -1;
	for (size_t i = 0; i < len; ++i) {
		if (h_out_naive[i] != h_out_optim[i]) {
			diff = i;
			break;
		}
	}

	std::cout << std::endl;
	if (diff == -1) {
		std::cout << "\x1b[1;32mtest passed!\x1b[0m\n" << std::endl;

		std::cout << "\x1b[1m\ttime\t\tthroughput\x1b[0m\n";
		std::cout
			<< "\x1b[1mhost\x1b[0m\t" << h_duration * 1.0e+3 << " ms\t"
			<< static_cast<double>(len * sizeof(int32_t)) / std::pow(2.0, 30) / h_duration << " GiB/s\n";
		std::cout
			<< "\x1b[1mdevice\x1b[0m\t" << d_duration * 1.0e+3 << " ms\t"
			<< static_cast<double>(len * sizeof(int32_t)) / std::pow(2.0, 30) / d_duration << " GiB/s\n";

		std::cout << "\n\x1b[1mdevice speedup:\x1b[0m\t" << h_duration / d_duration << "x\n";
	} else {
		std::cout << "\x1b[1;31mtest failed\x1b[0m" << std::endl;
		std::cout << "assertion failed at index: " << diff << "\n";

		ssize_t window_size = 10;
		std::cout << "\t\x1b[1mindex\t\thost\t\tdevice\x1b[0m\n";
		if (diff >= window_size / 2) {
			for (ssize_t i = -(window_size / 2); i < window_size / 2 + 1; ++i) {
				if (i == 0) {
					std::cout << "->\t\x1b[1;30;41m"
						<< diff + i << "\t\t"
						<< h_out_naive[diff + i] << "\t\t"
						<< h_out_optim[diff + i] << "\x1b[0m\n";
				} else {
					std::cout << "\t"
						<< diff + i << "\t\t"
						<< h_out_naive[diff + i] << "\t\t"
						<< h_out_optim[diff + i] << "\n";
				}
			}
		} else {
			for (ssize_t i = 0; i < window_size; ++i) {
				if (i == 0) {
					std::cout << "->\t\x1b[1;30;41m"
						<< diff + i << "\t\t"
						<< h_out_naive[diff + i] << "\t\t"
						<< h_out_optim[diff + i] << "\x1b[0m\n";
				} else {
					std::cout << "\t"
						<< diff + i << "\t\t"
						<< h_out_naive[diff + i] << "\t\t"
						<< h_out_optim[diff + i] << "\n";
				}
			}
		}
		std::cout << std::endl;
	}

	// Free host-side memory
	delete[] h_in;
	delete[] h_out_naive;
	delete[] h_out_optim;

	return 0;
}
