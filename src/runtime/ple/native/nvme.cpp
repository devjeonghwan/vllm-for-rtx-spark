// Native Windows ARM64 OS port of the pinned mbx_t_* file ABI.
// Format/MD5 admission recovered from original ELF PCs1c14..2610; no FP math.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <process.h>
#include <bcrypt.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#pragma comment(lib, "bcrypt.lib")

#define API extern "C" __declspec(dllexport)
static constexpr uint64_t PAGE_BYTES = 4096;
static constexpr const char* ALLOWED[] = {
    "a1f267476f175200e086a86b3d49c61c",
    "3bdcb62524a2662abc167bdcf2abea45",
    "cd721ff67a487be54279ebfbff877fde"
};

struct Handle {
    HANDLE value = INVALID_HANDLE_VALUE;
    Handle() = default;
    explicit Handle(HANDLE h) : value(h) {}
    ~Handle() { if (value && value != INVALID_HANDLE_VALUE) CloseHandle(value); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
};

static std::wstring wide(const char* s) {
    if (!s || !*s) throw std::runtime_error("empty UTF8 path");
    int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, nullptr, 0);
    if (!n) throw std::runtime_error("invalid UTF8 path");
    std::wstring result(static_cast<size_t>(n), L'\0');
    if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, result.data(), n))
        throw std::runtime_error("UTF8 conversion failed");
    result.resize(static_cast<size_t>(n - 1));
    if (result.size() < 3 || (result[1] != L':' && result.rfind(L"\\\\", 0) != 0))
        throw std::runtime_error("absolute path required");
    return result;
}

static uint64_t length(HANDLE h) {
    LARGE_INTEGER n{};
    if (!GetFileSizeEx(h, &n) || n.QuadPart < 0) throw std::runtime_error("file size");
    return static_cast<uint64_t>(n.QuadPart);
}

struct Layout {
    uint32_t packed, scales, stride, rows_per_page;
    Layout(int cp, int cs) {
        if (cp <= 0 || cs < 0 || cp > 4096 || cs > 4096 || cp + cs > 4096)
            throw std::runtime_error("column bounds");
        packed = static_cast<uint32_t>(cp); scales = static_cast<uint32_t>(cs);
        stride = std::max<uint32_t>(packed + scales, 96);
        rows_per_page = 4096 / stride;
    }
    uint64_t file_bytes(uint64_t rows) const {
        if (rows > (std::numeric_limits<uint64_t>::max() / 4096 - 1) * rows_per_page)
            throw std::runtime_error("file extent overflow");
        return ((rows + rows_per_page - 1) / rows_per_page) * 4096;
    }
};

// Buffered explicit-offset I/O, for source tensors / preparation only.
static bool transfer(HANDLE h, bool writing, uint64_t offset, void* data, size_t bytes) {
    auto* pointer = static_cast<unsigned char*>(data);
    while (bytes) {
        OVERLAPPED ov{};
        Handle event(CreateEventW(nullptr, TRUE, FALSE, nullptr));
        if (!event.value) return false;
        ov.hEvent = event.value;
        ov.Offset = static_cast<DWORD>(offset); ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
        DWORD wanted = static_cast<DWORD>(std::min<size_t>(bytes, 1U << 30)), completed = 0;
        BOOL ok = writing ? WriteFile(h, pointer, wanted, &completed, &ov)
                          : ReadFile(h, pointer, wanted, &completed, &ov);
        if (ok || GetLastError() == ERROR_IO_PENDING)
            ok = GetOverlappedResult(h, &ov, &completed, TRUE);
        if (!ok || completed == 0) return false;
        offset += completed; pointer += completed; bytes -= completed;
    }
    return true;
}

struct Md5 {
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    std::vector<unsigned char> object;
    Md5() {
        DWORD bytes = 0, used = 0;
        if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_MD5_ALGORITHM, nullptr, 0) < 0 ||
            BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&bytes), sizeof(bytes), &used, 0) < 0)
            throw std::runtime_error("MD5 provider");
        object.resize(bytes);
        if (BCryptCreateHash(algorithm, &hash, object.data(), bytes, nullptr, 0, 0) < 0)
            throw std::runtime_error("MD5 state");
    }
    ~Md5() { if (hash) BCryptDestroyHash(hash); if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0); }
    void update(const void* p, size_t n) {
        auto* bytes = static_cast<const unsigned char*>(p);
        while (n) {
            ULONG chunk = static_cast<ULONG>(std::min<size_t>(n, 1U << 30));
            if (BCryptHashData(hash, const_cast<PUCHAR>(bytes), chunk, 0) < 0) throw std::runtime_error("MD5 update");
            bytes += chunk; n -= chunk;
        }
    }
    std::string finish() {
        std::array<unsigned char, 16> bytes{};
        if (BCryptFinishHash(hash, bytes.data(), 16, 0) < 0) throw std::runtime_error("MD5 final");
        constexpr char hex[] = "0123456789abcdef";
        std::string out(32, '0');
        for (size_t i = 0; i < 16; ++i) { out[2*i] = hex[bytes[i] >> 4]; out[2*i+1] = hex[bytes[i] & 15]; }
        return out;
    }
};

struct PagePool {
    unsigned char* pages=nullptr;
    PagePool(){pages=static_cast<unsigned char*>(VirtualAlloc(nullptr,256*PAGE_BYTES,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE));
        if(!pages)throw std::bad_alloc();}
    ~PagePool(){if(pages)VirtualFree(pages,0,MEM_RELEASE);}
};
struct ReadBuffers {
    unsigned char* pages = nullptr;
    std::shared_ptr<PagePool> pool;
    std::vector<OVERLAPPED> operations;
    std::vector<int64_t> indices;
    std::vector<unsigned char> active;
    size_t pending = 0;
    ~ReadBuffers() { if (pages && !pool) VirtualFree(pages, 0, MEM_RELEASE); }
};

static constexpr size_t READ_CACHE_MAX_SLOTS = 256;
static constexpr ULONG COMPLETION_BATCH_MAX = 64;

struct ParallelLane;

struct FlatFile {
    Handle file, port;
    std::wstring path_name;
    std::mutex dispatch_mutex;
    std::shared_ptr<ParallelLane> lane;
    std::shared_ptr<PagePool> page_pool;
    std::shared_ptr<ReadBuffers> pooled_view,helper_view;
    uint64_t bytes;
    std::mutex mutex;
    std::vector<std::shared_ptr<ReadBuffers>> failed;
    std::shared_ptr<ReadBuffers> cached;
    explicit FlatFile(const std::wstring& path) : file(CreateFileW(path.c_str(), GENERIC_READ,
        FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING, nullptr)), bytes(0) {
        if (file.value == INVALID_HANDLE_VALUE) throw std::runtime_error("flat file open");
        path_name = path;
        bytes = length(file.value);
        if (!bytes || bytes % PAGE_BYTES) throw std::runtime_error("flat file page extent");
        port.value = CreateIoCompletionPort(file.value, nullptr, 0, 0);
        if (!port.value) throw std::runtime_error("IOCP create");
        failed.reserve(1); // Retaining the first uncertain I/O owner cannot allocate.
    }
};
static std::mutex files_mutex;
static std::unordered_map<int, std::shared_ptr<FlatFile>> files;
static std::atomic<int> next_file{1};
static thread_local unsigned queue_depth = 0;
static std::mutex signature_mutex;
static std::string last_signature;

// The file mutex serializes all use of its cache. Cached pages never own a file
// handle; closing the map releases its idle cache. Larger queue depths retain
// their original behavior, but their buffers are not kept between calls.
static std::shared_ptr<ReadBuffers> acquire_buffer(FlatFile& file, size_t slots) {
    if(file.pooled_view && slots<=file.pooled_view->operations.size()){
        if(file.pooled_view->pending)return nullptr;
        file.cached=file.pooled_view;return file.cached;
    }
    if (slots > READ_CACHE_MAX_SLOTS) {
        file.cached.reset();
    } else {
        if (file.cached && (file.cached.use_count() != 1 || file.cached->pending))
            file.cached.reset();
        if (file.cached && file.cached->operations.size() >= slots)
            return file.cached;
        file.cached.reset(); // Drop idle storage before allocating a larger cache.
    }
    auto memory = std::make_shared<ReadBuffers>();
    memory->operations.resize(slots);
    memory->indices.resize(slots);
    memory->active.resize(slots);
    memory->pages = static_cast<unsigned char*>(VirtualAlloc(nullptr, slots * PAGE_BYTES,
                                    MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (!memory->pages) return nullptr;
    if (slots <= READ_CACHE_MAX_SLOTS) file.cached = memory;
    return memory;
}

static bool pin_reader_module() noexcept {
    HMODULE module = nullptr;
    return GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
        reinterpret_cast<LPCWSTR>(&pin_reader_module), &module) != FALSE;
}

API int mbx_t_last_prepare_signature(char* output, int capacity) noexcept {
    if (!output || capacity < 33) return -EINVAL;
    std::lock_guard guard(signature_mutex);
    if (last_signature.size() != 32) { output[0] = 0; return 1; }
    std::memcpy(output, last_signature.c_str(), 33); return 0;
}

API int mbx_t_ok2(const char* path, int64_t rows, int cp, int cs) noexcept {
    try {
        if (rows < 0) return 0;
        Layout layout(cp, cs);
        Handle h(CreateFileW(wide(path).c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
        return h.value != INVALID_HANDLE_VALUE && length(h.value) == layout.file_bytes(static_cast<uint64_t>(rows));
    } catch (...) { return 0; }
}
API int mbx_t_ok(const char* path, int64_t rows) noexcept { return mbx_t_ok2(path, rows, 80, 16); }
API int mbx_t_fd(const char* path) noexcept {
    try {
        auto file = std::make_shared<FlatFile>(wide(path));
        std::lock_guard guard(files_mutex);
        int id = next_file.fetch_add(1);
        if (id <= 0) return -EMFILE;
        files.emplace(id, std::move(file)); return id;
    } catch (...) { return -EIO; }
}
API int mbx_t_open(int depth) noexcept {
    if (depth <= 0 || depth > 65536) return 1;
    queue_depth = static_cast<unsigned>(depth); return 0;
}

static int read_core(std::shared_ptr<FlatFile> file, const int64_t* rows, int64_t count, void* packed_output,
                   void* scales_output, int cp, int cs, unsigned requested_queue) noexcept {
    if (count <= 0) return 0;
    std::shared_ptr<ReadBuffers> memory;
    size_t pending = 0;
    try {
        Layout layout(cp, cs);
        if (!rows || !packed_output || (cs && !scales_output) || count > INT_MAX) return -EINVAL;
        std::lock_guard guard(file->mutex);
        if (!file->failed.empty()) return -EIO;
        for (int64_t i = 0; i < count; ++i)
            if (rows[i] < 0 || static_cast<uint64_t>(rows[i]) / layout.rows_per_page >= file->bytes / PAGE_BYTES) return -EINVAL;
        size_t slots = static_cast<size_t>(std::min<int64_t>(count, requested_queue));
        memory = acquire_buffer(*file, slots);
        if (!memory) return -ENOMEM;
        std::fill(memory->active.begin(), memory->active.begin() + slots, 0);
        memory->pending = 0;
        int64_t next = 0, finished = 0;
        bool failed = false, cancelled = false;
        auto retain = [&]() {
            pin_reader_module(); // Keep code/static owner alive if terminal proof is unknown.
            file->failed.push_back(memory);
            file->cached.reset();
            return -EIO;
        };
        auto submit = [&](size_t slot) {
            auto& ov = memory->operations[slot]; ov = {};
            uint64_t offset = static_cast<uint64_t>(rows[next]) / layout.rows_per_page * PAGE_BYTES;
            ov.Offset = static_cast<DWORD>(offset); ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
            memory->indices[slot] = next++;
            DWORD ignored = 0;
            BOOL ok = ReadFile(file->file.value, memory->pages + slot * PAGE_BYTES, PAGE_BYTES, &ignored, &ov);
            if (!ok && GetLastError() != ERROR_IO_PENDING) return false;
            memory->active[slot] = 1;
            memory->pending = ++pending;
            return true;
        };
        for (size_t i = 0; i < slots; ++i) if (!submit(i)) { failed = true; break; }
        if (failed && pending) { CancelIoEx(file->file.value, nullptr); cancelled = true; }
        while (pending) {
            OVERLAPPED_ENTRY entries[COMPLETION_BATCH_MAX]{};
            ULONG removed = 0;
            BOOL ok = GetQueuedCompletionStatusEx(file->port.value, entries,
                static_cast<ULONG>(std::min<size_t>(pending, COMPLETION_BATCH_MAX)), &removed, INFINITE, FALSE);
            if (!ok || !removed || removed > COMPLETION_BATCH_MAX) return retain();
            size_t completed[COMPLETION_BATCH_MAX]{};
            for (ULONG i = 0; i < removed; ++i) {
                auto* ov = entries[i].lpOverlapped;
                uintptr_t base = reinterpret_cast<uintptr_t>(memory->operations.data());
                uintptr_t address = reinterpret_cast<uintptr_t>(ov);
                if (address < base || address - base >= slots * sizeof(OVERLAPPED) ||
                    (address - base) % sizeof(OVERLAPPED) || entries[i].lpCompletionKey != 0)
                    return retain();
                size_t slot = (address - base) / sizeof(OVERLAPPED);
                if (!memory->active[slot]) return retain();

                // GQCSEx success only describes dequeuing. Check each request
                // using its original OVERLAPPED; OVERLAPPED_ENTRY.Internal is reserved.
                // https://learn.microsoft.com/windows/win32/api/ioapiset/nf-ioapiset-getoverlappedresult
                DWORD actual_bytes = 0;
                if (!GetOverlappedResult(file->file.value, ov, &actual_bytes, FALSE)) {
                    // A queued packet with a still-pending OVERLAPPED is not
                    // terminal proof. Preserve its active slot and DMA owner.
                    if (GetLastError() == ERROR_IO_INCOMPLETE) return retain();
                    failed = true;
                }
                memory->active[slot] = 0;
                memory->pending = --pending;
                completed[i] = slot;
                if (actual_bytes != PAGE_BYTES || entries[i].dwNumberOfBytesTransferred != PAGE_BYTES)
                    failed = true;
            }
            // Validate the complete dequeued batch before any refill can reuse
            // an OVERLAPPED. Duplicate or foreign packets cannot claim new work.
            if (!failed) {
                for (ULONG i = 0; i < removed; ++i) {
                    size_t slot = completed[i];
                    int64_t index = memory->indices[slot];
                    auto* row = memory->pages + slot * PAGE_BYTES +
                        (static_cast<uint64_t>(rows[index]) % layout.rows_per_page) * layout.stride;
                    std::memcpy(static_cast<unsigned char*>(packed_output) + static_cast<size_t>(index) * cp, row, cp);
                    if (cs) std::memcpy(static_cast<unsigned char*>(scales_output) + static_cast<size_t>(index) * cs, row + cp, cs);
                    ++finished;
                }
                for (ULONG i = 0; i < removed; ++i)
                    if (next < count && !submit(completed[i])) { failed = true; break; }
            }
            if (failed && !cancelled) { CancelIoEx(file->file.value, nullptr); cancelled = true; }
        }
        return !failed && finished == count ? 0 : -EIO;
    } catch (...) {
        if (pending && file && memory) {
            pin_reader_module(); // Preserve exception-retained code/static owners too.
            std::lock_guard guard(file->mutex);
            file->failed.push_back(memory);
            file->cached.reset();
        }
        return -EIO;
    }
}

struct ParallelLane {
    std::shared_ptr<FlatFile> child;
    Handle request, done, thread;
    std::atomic<bool> stop{false}, complete{true}, running{false}, poisoned{false};
    const int64_t* rows = nullptr;
    int64_t count = 0;
    void *packed = nullptr, *scales = nullptr;
    int cp = 0, cs = 0, result = -EIO;
    unsigned qd = 0;

    explicit ParallelLane(const std::wstring& path) : child(std::make_shared<FlatFile>(path)) {
        request.value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        done.value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!request.value || !done.value) throw std::runtime_error("lane events");
        if (!pin_reader_module()) throw std::runtime_error("lane module lifetime");
        uintptr_t worker = _beginthreadex(nullptr, 0, &ParallelLane::entry, this, 0, nullptr);
        if (!worker) throw std::runtime_error("lane thread");
        thread.value = reinterpret_cast<HANDLE>(worker);
    }
    ~ParallelLane() {
        // No destructor runs while a call is active: parent dispatch owns this lane.
        // A poisoned lane is retained in files, so uncertain DMA is not destroyed.
        stop.store(true, std::memory_order_release);
        if (thread.value) {
            SetEvent(request.value);
            for (;;) {
                DWORD code = STILL_ACTIVE;
                if (GetExitCodeThread(thread.value, &code) && code != STILL_ACTIVE) break;
                WaitForSingleObject(thread.value, 10);
            }
        }
    }
    static unsigned __stdcall entry(void* raw) noexcept {
        auto& lane = *static_cast<ParallelLane*>(raw);
        for (;;) {
            DWORD rc = WaitForSingleObject(lane.request.value, INFINITE);
            if (rc != WAIT_OBJECT_0) { lane.poisoned.store(true); break; }
            if (lane.stop.load(std::memory_order_acquire)) break;
            lane.running.store(true, std::memory_order_release);
            lane.result = read_core(lane.child, lane.rows, lane.count, lane.packed, lane.scales,
                                    lane.cp, lane.cs, lane.qd);
            lane.running.store(false, std::memory_order_release);
            lane.complete.store(true, std::memory_order_release);
            SetEvent(lane.done.value);
        }
        return 0;
    }
    bool unsafe() const noexcept {
        return poisoned.load() || !complete.load() || !child->failed.empty() ||
            (child->cached && child->cached->pending) || (child->pooled_view && child->pooled_view->pending);
    }
    bool start(const int64_t* r, int64_t n, void* p, void* s, int packed_cols, int scale_cols, unsigned queue) noexcept {
        if (unsafe()) return false;
        rows = r; count = n; packed = p; scales = s; cp = packed_cols; cs = scale_cols; qd = queue;
        result = -EIO;
        complete.store(false, std::memory_order_release);
        if (!SetEvent(request.value)) {
            poisoned.store(true);
            complete.store(true, std::memory_order_release); // No request published.
            return false;
        }
        return true;
    }
    int join() noexcept {
        while (!complete.load(std::memory_order_acquire)) {
            // Even a failed wait is not proof of completion. Keep caller buffers alive.
            WaitForSingleObject(done.value, 10);
        }
        return result;
    }
};

// Private stable shared scratch pool. Actual read_core queue limits remain unchanged.
static void clear_pool(const std::shared_ptr<FlatFile>& file) noexcept {
    file->cached.reset();
    if (file->lane) {
        auto& child=file->lane->child;
        child->cached.reset();child->pooled_view.reset();child->page_pool.reset();child->helper_view.reset();
    }
    file->pooled_view.reset();file->helper_view.reset();file->page_pool.reset();
}
static std::shared_ptr<ReadBuffers> pool_view(const std::shared_ptr<PagePool>& pool,size_t offset,size_t slots) {
    auto view=std::make_shared<ReadBuffers>();
    view->operations.resize(slots);view->indices.resize(slots);view->active.resize(slots);
    view->pool=pool;view->pages=pool->pages+offset*PAGE_BYTES;
    return view;
}
static bool ensure_pool(const std::shared_ptr<FlatFile>& file) {
    if (file->page_pool) return true;
    if (!pin_reader_module()) return false;
    // Release legacy idle allocation before creating the one physical pool.
    clear_pool(file);
    auto pool=std::make_shared<PagePool>();
    auto primary=pool_view(pool,0,256),helper=pool_view(pool,128,128);
    file->page_pool=pool;file->pooled_view=primary;file->helper_view=helper;file->cached=primary;
    if(file->lane){file->lane->child->page_pool=pool;file->lane->child->pooled_view=helper;file->lane->child->cached=helper;}
    return true;
}
static bool partition_idle_cache(const std::shared_ptr<FlatFile>& file,bool) {
    if ((file->cached&&file->cached->pending)||(file->pooled_view&&file->pooled_view->pending)||
        (file->lane&&file->lane->unsafe())) return false;
    return ensure_pool(file);
}
API int mbx_t_read(int descriptor, const int64_t* rows, int64_t count, void* packed_output,
                   void* scales_output, int cp, int cs, int threads) noexcept {
    if (count <= 0) return 0;
    try {
        Layout layout(cp, cs);
        if (!rows || !packed_output || (cs && !scales_output) || count > INT_MAX) return -EINVAL;
        std::shared_ptr<FlatFile> file;
        { std::lock_guard list(files_mutex); auto it = files.find(descriptor); if (it == files.end()) return -EBADF; file = it->second; }
        std::lock_guard dispatch(file->dispatch_mutex);
        unsigned qd = queue_depth ? queue_depth : unsigned(std::max(threads,1));
        if (!file->failed.empty() || (file->lane && file->lane->unsafe())) return -EIO;
        if (qd != 256 || count < 512) {
            if (!partition_idle_cache(file,false)) return -EIO;
            return read_core(file,rows,count,packed_output,scales_output,cp,cs,qd);
        }
        // Preserve original all-input validation before any lane can write output.
        for (int64_t i = 0; i < count; ++i)
            if (rows[i] < 0 || static_cast<uint64_t>(rows[i]) / layout.rows_per_page >= file->bytes / PAGE_BYTES) return -EINVAL;
        if (!partition_idle_cache(file,true)) return -EIO;
        if (!file->lane) file->lane = std::make_shared<ParallelLane>(file->path_name);
        auto& lane = *file->lane;
        lane.child->page_pool=file->page_pool;lane.child->pooled_view=file->helper_view;lane.child->cached=file->helper_view;
        int64_t first = (count + 1) / 2, second = count - first;
        unsigned primary_qd = (qd + 1) / 2, helper_qd = qd / 2;
        if (!lane.start(rows+first,second,static_cast<unsigned char*>(packed_output)+size_t(first)*cp,
              cs ? static_cast<unsigned char*>(scales_output)+size_t(first)*cs : nullptr,cp,cs,helper_qd)) return -EIO;
        int primary = read_core(file,rows,first,packed_output,scales_output,cp,cs,primary_qd);
        int helper = lane.join(); // Always join even when the first lane failed.
        return primary ? primary : helper;
    } catch (...) { return -EIO; }
}


API int mbx_t_close(int descriptor) noexcept {
    try {
        std::lock_guard guard(files_mutex);
        auto it = files.find(descriptor); if (it == files.end()) return 0;
        auto owner = it->second;
        std::lock_guard dispatch(owner->dispatch_mutex);
        std::lock_guard io_guard(owner->mutex);
        if (!owner->failed.empty() || (owner->cached && owner->cached->pending) ||
            (owner->pooled_view && owner->pooled_view->pending) ||
            (owner->lane && owner->lane->unsafe())) return -EBUSY;
        owner->lane.reset();
        owner->cached.reset();
        files.erase(it); return 0;
    } catch (...) { return -EIO; }
}

API int mbx_t_prepare(const char* const* paths, const int64_t* offsets, int shards,
                      int64_t rows_per_shard, int cp, int cs, const char* output) noexcept {
    std::wstring temporary;
    { std::lock_guard guard(signature_mutex); last_signature.clear(); }
    try {
        Layout layout(cp, cs);
        if (!paths || !offsets || shards <= 0 || shards > 65536 || rows_per_shard <= 0 ||
            static_cast<uint64_t>(rows_per_shard) > UINT64_MAX / static_cast<unsigned>(shards)) return -EINVAL;
        const uint64_t total = static_cast<uint64_t>(rows_per_shard) * shards;
        std::vector<std::unique_ptr<Handle>> input;
        std::vector<std::unique_ptr<Md5>> hashes;
        for (int i = 0; i < shards * 2; ++i) {
            auto h = std::make_unique<Handle>(CreateFileW(wide(paths[i]).c_str(), GENERIC_READ, FILE_SHARE_READ,
                nullptr, OPEN_EXISTING, FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN, nullptr));
            uint64_t columns = (i & 1) ? cs : cp;
            if (h->value == INVALID_HANDLE_VALUE || offsets[i] < 0 ||
                static_cast<uint64_t>(rows_per_shard) > (UINT64_MAX - static_cast<uint64_t>(offsets[i])) / std::max<uint64_t>(columns, 1) ||
                length(h->value) < static_cast<uint64_t>(offsets[i]) + static_cast<uint64_t>(rows_per_shard) * columns) return -EIO;
            input.push_back(std::move(h)); hashes.push_back(std::make_unique<Md5>());
        }
        std::wstring target = wide(output); temporary = target + L".tmp";
        {
            Handle output_handle(CreateFileW(temporary.c_str(), GENERIC_READ | GENERIC_WRITE, 0,
                nullptr, CREATE_ALWAYS, FILE_FLAG_OVERLAPPED, nullptr));
            if (output_handle.value == INVALID_HANDLE_VALUE) return -EIO;
            const uint64_t batch_capacity = static_cast<uint64_t>(layout.rows_per_page) * 24000; // original1df8
            for (uint64_t start = 0; start < total; start += batch_capacity) {
                uint64_t count = std::min(batch_capacity, total - start);
                std::vector<unsigned char> packed(static_cast<size_t>(count) * cp), scales(static_cast<size_t>(count) * cs);
                for (int plane = 0; plane < 2; ++plane) {
                    size_t columns = plane ? cs : cp;
                    auto& buffer = plane ? scales : packed;
                    uint64_t cursor = start;
                    while (cursor < start + count) {
                        uint64_t shard = cursor / rows_per_shard, local = cursor % rows_per_shard;
                        uint64_t take = std::min<uint64_t>(rows_per_shard - local, start + count - cursor);
                        size_t source = static_cast<size_t>(shard * 2 + plane);
                        void* destination = columns ? buffer.data() + static_cast<size_t>(cursor - start) * columns : nullptr;
                        size_t bytes = static_cast<size_t>(take) * columns;
                        if (bytes && !transfer(input[source]->value, false, static_cast<uint64_t>(offsets[source]) + local * columns, destination, bytes)) return 1;
                        hashes[source]->update(destination, bytes); cursor += take;
                    }
                }
                std::vector<unsigned char> pages(static_cast<size_t>(layout.file_bytes(count)), 0);
                for (uint64_t row = 0; row < count; ++row) {
                    size_t off = static_cast<size_t>(row / layout.rows_per_page * 4096 + row % layout.rows_per_page * layout.stride);
                    std::memcpy(pages.data() + off, packed.data() + row * cp, cp);
                    if (cs) std::memcpy(pages.data() + off + cp, scales.data() + row * cs, cs);
                }
                if (!transfer(output_handle.value, true, start / layout.rows_per_page * 4096, pages.data(), pages.size())) return 2;
                if (!FlushFileBuffers(output_handle.value)) return 2;
            }
            // Original fixed xorshift seed; compare4096 source rows against the flat map.
            uint64_t rng = 0x9e3779b97f4a7c15ULL;
            std::array<unsigned char, 4096> page{};
            std::vector<unsigned char> row(static_cast<size_t>(cp + cs));
            for (int i = 0; i < 4096; ++i) {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                uint64_t id = rng % total, shard = id / rows_per_shard, local = id % rows_per_shard;
                if (!transfer(output_handle.value, false, id / layout.rows_per_page * 4096, page.data(), page.size())) return 3;
                if (!transfer(input[shard * 2]->value, false, offsets[shard * 2] + local * cp, row.data(), cp) ||
                    (cs && !transfer(input[shard * 2 + 1]->value, false, offsets[shard * 2 + 1] + local * cs, row.data() + cp, cs))) return 3;
                if (std::memcmp(page.data() + id % layout.rows_per_page * layout.stride, row.data(), row.size())) return 4;
            }
            Md5 aggregate;
            for (auto& h : hashes) { auto text = h->finish(); aggregate.update(text.data(), text.size()); }
            auto signature = aggregate.finish();
            { std::lock_guard guard(signature_mutex); last_signature = signature; }
            bool known = false;
            for (auto expected : ALLOWED) known |= signature == expected;
            if (!known) return 5;
        }
        if (!MoveFileExW(temporary.c_str(), target.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) return -EIO;
        return 0;
    } catch (const std::bad_alloc&) { return -ENOMEM; }
      catch (...) { return -EINVAL; }
    // Temporary-file cleanup is performed by the caller only after inspecting
    // failed preparation; no original checkpoint path is ever opened writable.
}
