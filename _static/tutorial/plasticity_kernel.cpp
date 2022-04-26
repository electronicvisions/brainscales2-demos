#include "libnux/vx/time.h"
#include "libnux/vx/vector.h"
#include <array>
#include <cstdint>

constexpr static size_t image_h = 64;
constexpr static size_t image_w = 64;

/**
 * Image data to be set from outside.
 */
uint8_t volatile image[image_h][image_w];

/**
 * Wait duration between weight changes [PPU cycles].
 */
uint32_t volatile wait_duration;

constexpr static size_t row = 1;

int start()
{
	for (size_t h = 0; h < image_h; ++h) {
		// get current time
		libnux::vx::time_base_t const before = libnux::vx::now();

		// load new vertical line of image
		libnux::vx::vector_row_t weight_row;
		for (size_t w = 0; w < image_w; ++w) {
			weight_row[w] = image[h][w];
		}

		// alter weights in synapse memory
		libnux::vx::set_row_via_vector(weight_row, row, libnux::vx::dls_weight_base);

		// wait until wait_duration elapsed
		while (libnux::vx::now() < (before + static_cast<libnux::vx::time_base_t>(wait_duration))) {
		}
	}
	return 42;
}
