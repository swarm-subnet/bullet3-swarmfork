#ifndef B3_DISK_CACHE_H
#define B3_DISK_CACHE_H

// Files kept under SWARM_BVH_CACHE_DIR, the folder a validator gives each epoch, so work that depends only on bytes
// read from disk is done once per machine. A file is found by a key hashed from those bytes, never by a name, so a
// changed source can only miss; it lands through a rename, so a reader never sees half of one.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#ifdef _WIN32
#include <process.h>
#define b3DiskCacheProcessId _getpid
#else
#include <unistd.h>
#define b3DiskCacheProcessId getpid
#endif

// 64-bit hash of a byte run, eight bytes at a time; chained through `hash` to cover several runs.
inline unsigned long long b3DiskCacheHash(const void* data, size_t length, unsigned long long hash = 14695981039346656037ULL)
{
	const unsigned char* bytes = (const unsigned char*)data;
	size_t i = 0;
	for (; i + 8 <= length; i += 8)
	{
		unsigned long long word;
		memcpy(&word, bytes + i, 8);
		hash = (hash ^ word) * 1099511628211ULL;
		hash ^= hash >> 29;
	}
	for (; i < length; i++)
		hash = (hash ^ bytes[i]) * 1099511628211ULL;
	return hash ^ (unsigned long long)length;
}

// The path of the cache file for `key` with the kind's suffix; false when no cache folder is set.
inline bool b3DiskCachePath(unsigned long long key, const char* suffix, std::string& path)
{
	const char* dir = getenv("SWARM_BVH_CACHE_DIR");
	if (!dir || !*dir)
		return false;
	char name[1200];
	snprintf(name, sizeof(name), "%s/%016llx.%s", dir, key, suffix);
	path = name;
	return true;
}

// The whole file at `path`; false when it cannot be read whole.
inline bool b3DiskCacheRead(const std::string& path, std::vector<char>& out)
{
	FILE* f = fopen(path.c_str(), "rb");
	if (!f)
		return false;
	bool ok = fseek(f, 0, SEEK_END) == 0;
	const long size = ok ? ftell(f) : -1;
	ok = ok && size >= 0 && fseek(f, 0, SEEK_SET) == 0;
	if (ok)
	{
		out.resize((size_t)size);
		ok = size == 0 || fread(&out[0], 1, (size_t)size, f) == (size_t)size;
	}
	fclose(f);
	return ok;
}

// Writes `length` bytes to `path` through a file of this process's own, renamed into place when whole.
inline void b3DiskCacheWrite(const std::string& path, const void* data, size_t length)
{
	char tmp[1300];
	snprintf(tmp, sizeof(tmp), "%s.%d.tmp", path.c_str(), (int)b3DiskCacheProcessId());
	FILE* f = fopen(tmp, "wb");
	if (!f)
		return;
	bool ok = length == 0 || fwrite(data, 1, length, f) == length;
	ok = (fclose(f) == 0) && ok;
	if (!ok || rename(tmp, path.c_str()) != 0)
		remove(tmp);
}

// Appends and reads back plain values and blocks in one byte buffer, for the cache files' simple layouts.
struct b3DiskCacheWriter
{
	std::vector<char> m_bytes;
	void put(const void* data, size_t length)
	{
		m_bytes.insert(m_bytes.end(), (const char*)data, (const char*)data + length);
	}
	template <typename T>
	void value(const T& v)
	{
		put(&v, sizeof(T));
	}
	void text(const std::string& s)
	{
		value((unsigned long long)s.size());
		put(s.data(), s.size());
	}
	template <typename T>
	void block(const std::vector<T>& v)
	{
		value((unsigned long long)v.size());
		if (!v.empty())
			put(&v[0], v.size() * sizeof(T));
	}
};

struct b3DiskCacheReader
{
	const char* m_at;
	const char* m_end;
	bool m_ok;
	b3DiskCacheReader(const std::vector<char>& bytes) : m_at(bytes.empty() ? 0 : &bytes[0]), m_end(m_at + bytes.size()), m_ok(true) {}
	bool take(void* out, size_t length)
	{
		m_ok = m_ok && (size_t)(m_end - m_at) >= length;
		if (m_ok && length)
		{
			memcpy(out, m_at, length);
			m_at += length;
		}
		return m_ok;
	}
	template <typename T>
	bool value(T& v)
	{
		return take(&v, sizeof(T));
	}
	bool text(std::string& s)
	{
		unsigned long long n = 0;
		if (!value(n) || n > (unsigned long long)(m_end - m_at))
			return m_ok = false;
		s.assign(m_at, (size_t)n);
		m_at += n;
		return true;
	}
	template <typename T>
	bool block(std::vector<T>& v)
	{
		unsigned long long n = 0;
		if (!value(n) || n > (unsigned long long)(m_end - m_at) / sizeof(T))
			return m_ok = false;
		v.resize((size_t)n);
		return n == 0 || take(&v[0], (size_t)n * sizeof(T));
	}
};

#endif  // B3_DISK_CACHE_H
