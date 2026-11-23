#ifndef SWARM_FRAME_TEAM_H
#define SWARM_FRAME_TEAM_H

#include <atomic>
#include <exception>
#include <thread>
#include <xmmintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// One team of render threads for a whole frame. A parallel region per pass wakes the sleeping helpers every time and
// makes the caller wait at its end for helpers that may not have started yet; a frame team wakes them once, keeps them
// spinning between the passes, and the caller waits only for work a helper has already taken. Every pass hands its
// work out to whichever thread asks next, so the bytes never depend on which thread takes what.
class SwarmFrameTeam
{
public:
	// Work a thread takes at a time on a pass over a frame's rows or its pixels, small enough to share with a late helper.
	static const int kRowChunk = 8;
	static const int kPixelChunk = 8192;

	// Runs body on the calling thread with up to threads - 1 helpers beside it, which take part in every share and
	// forChunks call body makes on this thread. The helpers spin only until body returns or throws.
	template <class Body>
	static void run(int threads, const Body& body)
	{
#ifdef _OPENMP
		// Spinning helpers past the processors this thread may run on would only take turns with the caller.
		const int processors = omp_get_num_procs();
		const int size = threads < processors ? threads : processors;
		if (size > 1 && !current())
		{
			SwarmFrameTeam team;
			std::exception_ptr failure;
#pragma omp parallel num_threads(size)
			{
				if (omp_get_thread_num() == 0)
				{
					current() = &team;
					try
					{
						body();
					}
					catch (...)
					{
						failure = std::current_exception();
					}
					current() = 0;
					team.m_done.store(true);
					// The caller reaches the region's closing barrier after every awake helper, so it never sleeps there.
					unsigned spins = 0;
					while (team.m_awake.load() != 0)
						idle(spins);
				}
				else
					team.serve(omp_get_thread_num());
			}
			if (failure)
				std::rethrow_exception(failure);
			return;
		}
#endif
		body();
	}

	// Runs fn(worker) on the calling thread and on every helper that comes while the caller's own call runs; worker is
	// the thread's index in the team, below threads. fn hands its work out itself, so a helper that comes late finds
	// none left. Outside run it opens a parallel region of threads for fn, as a pass did before.
	template <class Fn>
	static void share(int threads, const Fn& fn)
	{
		SwarmFrameTeam* team = current();
		if (team && !team->m_sharing)
		{
			team->open(&invoke<Fn>, &fn);
			team->call(0);
			team->close();
			return;
		}
		// A share inside a share runs on this thread alone, as a nested parallel region did.
		if (team)
		{
			fn(0);
			return;
		}
#ifdef _OPENMP
		if (threads > 1)
		{
#pragma omp parallel num_threads(threads)
			fn(omp_get_thread_num());
			return;
		}
#endif
		fn(0);
	}

	// fn(chunk, worker) for every chunk below numChunks, each taken by whichever thread asks next.
	template <class Fn>
	static void forChunks(int threads, long long numChunks, const Fn& fn)
	{
		if (numChunks <= 0)
			return;
		std::atomic<long long> next(0);
		share(threads, [&](int worker) {
			for (long long chunk = next++; chunk < numChunks; chunk = next++)
				fn(chunk, worker);
		});
	}

	// fn(i) for every i below count, chunk indices at a time. fn is copied for each chunk, so what it captures by value
	// stays in registers through the loop.
	template <class Fn>
	static void forEach(int threads, int count, int chunk, const Fn& fn)
	{
		forChunks(threads, (count + (long long)chunk - 1) / chunk, [&](long long c, int) {
			const Fn local(fn);
			const int begin = (int)c * chunk;
			const int end = count - begin > chunk ? begin + chunk : count;
			for (int i = begin; i < end; i++)
				local(i);
		});
	}

private:
	// Odd while a job is open; helpers count themselves in before they read it.
	std::atomic<unsigned> m_job;
	std::atomic<int> m_inside;
	std::atomic<int> m_awake;
	std::atomic<bool> m_done;
	std::atomic<bool> m_failed;
	// Whether the caller has a job open; only the caller reads or writes it.
	bool m_sharing;
	std::exception_ptr m_failure;
	void (*m_run)(const void*, int);
	const void* m_fn;

	SwarmFrameTeam() : m_job(0), m_inside(0), m_awake(0), m_done(false), m_failed(false), m_sharing(false), m_run(0), m_fn(0) {}

	// The team the calling thread runs a frame in, if any; a thread of its own for every render call.
	static SwarmFrameTeam*& current()
	{
		static thread_local SwarmFrameTeam* team = 0;
		return team;
	}

	// A pause each turn and now and then the processor, so a team with more threads than cores lets the busy ones run.
	static void idle(unsigned& spins)
	{
		_mm_pause();
		if (++spins % 128 == 0)
			std::this_thread::yield();
	}

	template <class Fn>
	static void invoke(const void* fn, int worker)
	{
		(*static_cast<const Fn*>(fn))(worker);
	}

	void call(int worker)
	{
		try
		{
			m_run(m_fn, worker);
		}
		catch (...)
		{
			bool expected = false;
			if (m_failed.compare_exchange_strong(expected, true))
				m_failure = std::current_exception();
		}
	}

	void open(void (*run)(const void*, int), const void* fn)
	{
		m_run = run;
		m_fn = fn;
		m_sharing = true;
		m_job.store(m_job.load(std::memory_order_relaxed) + 1);
	}

	// Waits only for helpers already inside the job: they are awake and each finishes what it took.
	void close()
	{
		m_job.store(m_job.load(std::memory_order_relaxed) + 1);
		unsigned spins = 0;
		while (m_inside.load() != 0)
			idle(spins);
		m_sharing = false;
		if (m_failed.load(std::memory_order_relaxed))
		{
			m_failed.store(false, std::memory_order_relaxed);
			std::exception_ptr failure;
			failure.swap(m_failure);
			std::rethrow_exception(failure);
		}
	}

	// A helper's frame: join each new job once, spin between jobs, leave when the frame is done.
	void serve(int worker)
	{
		m_awake.fetch_add(1);
		unsigned seen = 0, spins = 0;
		for (;;)
		{
			const unsigned job = m_job.load(std::memory_order_acquire);
			if ((job & 1) && job != seen)
			{
				seen = job;
				m_inside.fetch_add(1);
				// Checked again once counted in, so the job cannot close and be replaced while this helper reads it.
				if (m_job.load() == job)
					call(worker);
				m_inside.fetch_sub(1, std::memory_order_release);
			}
			else if (m_done.load())
			{
				m_awake.fetch_sub(1, std::memory_order_release);
				return;
			}
			else
				idle(spins);
		}
	}
};

#endif  // SWARM_FRAME_TEAM_H
