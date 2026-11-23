#include "SwarmFrameTeam.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>

// A pass started inside another pass runs on its own thread alone, and every chunk of both runs exactly once.
int main()
{
	const int outerChunks = 64, innerChunks = 16;
	for (int threads = 1; threads <= 4; threads++)
	{
		for (int round = 0; round < 200; round++)
		{
			static std::atomic<int> outer[64], inner[64 * 16], wrongWorker;
			for (int i = 0; i < outerChunks; i++)
				outer[i] = 0;
			for (int i = 0; i < outerChunks * innerChunks; i++)
				inner[i] = 0;
			wrongWorker = 0;
			SwarmFrameTeam::run(threads, [&]() {
				SwarmFrameTeam::forChunks(threads, outerChunks, [&](long long c, int) {
					outer[c]++;
					SwarmFrameTeam::forChunks(threads, innerChunks, [&](long long d, int worker) {
						if (worker != 0)
							wrongWorker++;
						inner[c * innerChunks + d]++;
					});
				});
			});
			for (int i = 0; i < outerChunks; i++)
				if (outer[i] != 1) return EXIT_FAILURE;
			for (int i = 0; i < outerChunks * innerChunks; i++)
				if (inner[i] != 1) return EXIT_FAILURE;
			if (wrongWorker != 0) return EXIT_FAILURE;
		}
	}
	std::puts("Frame team nesting: PASS (1 to 4 threads, 200 rounds each)");
	return EXIT_SUCCESS;
}
