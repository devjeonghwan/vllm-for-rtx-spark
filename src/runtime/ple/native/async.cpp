// Native Windows ARM64 OS port of original mbx_pa_* SPSC stream-flag protocol.
// The original Python forward/post/wait and NVFP4 math remain unchanged.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <process.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#define API extern "C" __declspec(dllexport)
using Result = int;
using Stream = void*;
using Context = void*;
using DeviceAddress = unsigned long long;
using Write32 = Result(__stdcall*)(Stream, DeviceAddress, unsigned, unsigned);
using Wait32 = Result(__stdcall*)(Stream, DeviceAddress, unsigned, unsigned);
using HostAlias = Result(__stdcall*)(DeviceAddress*, void*, unsigned);
using CurrentContext = Result(__stdcall*)(Context*);
using ReadRows = int(*)(int, const int64_t*, int64_t, void*, void*, int, int, int);
using OpenQueue = int(*)(int);

struct State {
    int descriptor, cp, cs, maximum, threads, queue;
    DWORD owner = GetCurrentThreadId();
    int64_t* ids;
    unsigned char *packed, *scales;
    uint32_t *request, *done;
    DeviceAddress request_device = 0, done_device = 0;
    Context context = nullptr;
    HMODULE cuda = nullptr, reader = nullptr;
    Write32 write = nullptr; Wait32 wait = nullptr; HostAlias alias = nullptr;
    CurrentContext current = nullptr;
    ReadRows read = nullptr; OpenQueue open = nullptr;
    HANDLE worker = nullptr;
    std::atomic<bool> stop{false}, busy{false}, closing{false};
    std::atomic<int> failed{0};
    std::atomic<int64_t> jobs{0};
    std::atomic<double> read_us{0.0};
    std::vector<std::pair<int64_t, int>> order;
    std::vector<int64_t> unique;
    std::vector<int> inverse;
    std::vector<unsigned char> unique_packed, unique_scales;
};
static std::mutex state_mutex;
static State* state = nullptr; // process-global as in the original ABI

static uint32_t load(uint32_t* p) noexcept {
    return std::atomic_ref<uint32_t>(*p).load(std::memory_order_acquire);
}
static void store(uint32_t* p, uint32_t v) noexcept {
    std::atomic_ref<uint32_t>(*p).store(v, std::memory_order_release);
}
static std::wstring wide(const char* s) {
    if (!s || !*s) return {};
    int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, nullptr, 0);
    if (!n) return {};
    std::wstring result(static_cast<size_t>(n), L'\0');
    if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, result.data(), n)) return {};
    result.resize(static_cast<size_t>(n - 1));
    if (result.size() < 3 || (result[1] != L':' && result.rfind(L"\\\\", 0) != 0)) return {};
    return result;
}
static FARPROC symbol(HMODULE h, const char* first, const char* legacy = nullptr) {
    auto result = GetProcAddress(h, first);
    return result || !legacy ? result : GetProcAddress(h, legacy);
}

#ifdef MBX_PLE_CPU_TEST
// Explicit test-only Driver stand-ins. The production DLL has no such branch.
static Result __stdcall cpu_write(Stream, DeviceAddress p, unsigned value, unsigned flags) {
    if (flags) return 1;
    store(reinterpret_cast<uint32_t*>(p), value); return 0;
}
static Result __stdcall cpu_wait(Stream, DeviceAddress p, unsigned value, unsigned flags) {
    if (flags != 1) return 1;
    auto end = GetTickCount64() + 10000;
    while (load(reinterpret_cast<uint32_t*>(p)) != value) {
        if (GetTickCount64() >= end) return 2;
        SwitchToThread();
    }
    return 0;
}
static Result __stdcall cpu_alias(DeviceAddress* out, void* p, unsigned flags) {
    if (flags) return 1;
    *out = reinterpret_cast<DeviceAddress>(p); return 0;
}
static Result __stdcall cpu_current(Context* out) { *out = reinterpret_cast<Context>(1); return 0; }
#endif

API int mbx_pa_cpu_fixture_mode() noexcept {
#ifdef MBX_PLE_CPU_TEST
    return 1;
#else
    return 0;
#endif
}

// Original polling:201 probes then20us/200us/1ms according to idle50ms/2s.
// A process-local high-resolution waitable timer changes no system timer settings.
static unsigned __stdcall worker(void* pointer) noexcept {
    State& a = *static_cast<State*>(pointer);
    if (a.open && a.open(a.queue) == 0) a.threads = 0;
    HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, 0x2, TIMER_ALL_ACCESS);
    if (!timer) timer = CreateWaitableTimerW(nullptr, TRUE, nullptr);
    auto last = std::chrono::steady_clock::now();
    while (!a.stop.load(std::memory_order_acquire)) {
        unsigned count = 0;
        for (int probe = 0; probe < 201 && !(count = load(a.request)); ++probe) YieldProcessor();
        if (!count) {
            auto idle = std::chrono::steady_clock::now() - last;
            int64_t us = idle < std::chrono::milliseconds(50) ? 20 : idle < std::chrono::seconds(2) ? 200 : 1000;
            LARGE_INTEGER due{}; due.QuadPart = -us * 10;
            if (timer && SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) WaitForSingleObject(timer, INFINITE);
            else SwitchToThread();
            continue;
        }
        a.busy.store(true, std::memory_order_release);
        store(a.request, 0); // original acquire -> request0 before I/O
        if (count > static_cast<unsigned>(a.maximum)) { a.failed.store(2); count = static_cast<unsigned>(a.maximum); }
        const auto start = std::chrono::steady_clock::now();
        for (unsigned i = 0; i < count; ++i) a.order[i] = {a.ids[i], static_cast<int>(i)};
        std::sort(a.order.begin(), a.order.begin() + count,
                  [](const auto& x, const auto& y) { return x.first < y.first; });
        size_t distinct = 0;
        for (unsigned i = 0; i < count; ++i) {
            if (!distinct || a.unique[distinct - 1] != a.order[i].first) a.unique[distinct++] = a.order[i].first;
            a.inverse[static_cast<size_t>(a.order[i].second)] = static_cast<int>(distinct - 1);
        }
        int rc = a.read(a.descriptor, a.unique.data(), static_cast<int64_t>(distinct),
                        a.unique_packed.data(), a.unique_scales.data(), a.cp, a.cs, a.threads);
        if (rc) {
            a.failed.store(1, std::memory_order_release); // sticky, original Python tests truthiness
            std::memset(a.packed, 0, static_cast<size_t>(count) * a.cp);
            if (a.cs) std::memset(a.scales, 0, static_cast<size_t>(count) * a.cs);
        } else {
            for (unsigned i = 0; i < count; ++i) {
                size_t source = static_cast<size_t>(a.inverse[i]);
                std::memcpy(a.packed + static_cast<size_t>(i) * a.cp, a.unique_packed.data() + source * a.cp, a.cp);
                if (a.cs) std::memcpy(a.scales + static_cast<size_t>(i) * a.cs, a.unique_scales.data() + source * a.cs, a.cs);
            }
        }
        a.read_us.fetch_add(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count());
        a.jobs.fetch_add(1);
        // No sequence counter. Failure also publishes done1 to unblock GPU wait.
        store(a.done, 1);
        a.busy.store(false, std::memory_order_release);
        last = std::chrono::steady_clock::now();
    }
    if (timer) CloseHandle(timer);
    return 0;
}

API int mbx_pa_init(const char* reader_path, int descriptor, int cp, int cs, int maximum,
                    int threads, int queue, uint64_t ids, uint64_t packed, uint64_t scales,
                    uint64_t request, uint64_t done) noexcept {
    std::lock_guard guard(state_mutex);
    if (state) {
        // Do not silently bind a second Torch allocation to the original singleton.
        return state->descriptor == descriptor && state->cp == cp && state->cs == cs &&
               state->maximum == maximum && reinterpret_cast<uint64_t>(state->ids) == ids &&
               reinterpret_cast<uint64_t>(state->packed) == packed && reinterpret_cast<uint64_t>(state->scales) == scales &&
               reinterpret_cast<uint64_t>(state->request) == request && reinterpret_cast<uint64_t>(state->done) == done ? 0 : 1;
    }
    if (cp <= 0 || cs < 0 || cp + cs > 4096 || maximum <= 0 || !ids || !packed || (cs && !scales) ||
        !request || !done || request % 4 || done % 4 || request == done) return 1;
    State* a = nullptr;
    try {
        auto path = wide(reader_path); if (path.empty()) return 2;
        a = new State();
        a->descriptor = descriptor; a->cp = cp; a->cs = cs; a->maximum = maximum;
        a->threads = std::max(threads, 1); a->queue = queue;
        a->ids = reinterpret_cast<int64_t*>(ids); a->packed = reinterpret_cast<unsigned char*>(packed);
        a->scales = reinterpret_cast<unsigned char*>(scales);
        a->request = reinterpret_cast<uint32_t*>(request); a->done = reinterpret_cast<uint32_t*>(done);
        a->reader = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!a->reader) throw 3;
#ifdef MBX_PLE_CPU_TEST
        a->write = cpu_write; a->wait = cpu_wait; a->alias = cpu_alias; a->current = cpu_current;
#else
        a->cuda = LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!a->cuda) throw 3;
        a->write = reinterpret_cast<Write32>(symbol(a->cuda, "cuStreamWriteValue32_v2", "cuStreamWriteValue32"));
        a->wait = reinterpret_cast<Wait32>(symbol(a->cuda, "cuStreamWaitValue32_v2", "cuStreamWaitValue32"));
        a->alias = reinterpret_cast<HostAlias>(symbol(a->cuda, "cuMemHostGetDevicePointer_v2"));
        a->current = reinterpret_cast<CurrentContext>(symbol(a->cuda, "cuCtxGetCurrent"));
#endif
        a->read = reinterpret_cast<ReadRows>(symbol(a->reader, "mbx_t_read"));
        a->open = reinterpret_cast<OpenQueue>(symbol(a->reader, "mbx_t_open"));
        if (!a->write || !a->wait || !a->alias || !a->current || !a->read) throw 4;
        if (a->current(&a->context) || !a->context) throw 5;
        // Original arguments are CPU pointers. Query the actual Windows CUDA aliases.
        if (a->alias(&a->request_device, a->request, 0) || a->alias(&a->done_device, a->done, 0)) throw 6;
        a->order.resize(maximum); a->unique.resize(maximum); a->inverse.resize(maximum);
        a->unique_packed.resize(static_cast<size_t>(maximum) * cp);
        a->unique_scales.resize(std::max<size_t>(static_cast<size_t>(maximum) * cs, 1));
        store(a->request, 0); store(a->done, 0);
        // Original Python retains its library globally. Pin while a native worker
        // exists so no unload can strand code/pinned-memory references.
        HMODULE self = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                               reinterpret_cast<LPCWSTR>(&mbx_pa_init), &self)) throw 8;
        a->worker = reinterpret_cast<HANDLE>(_beginthreadex(nullptr, 0, worker, a, 0, nullptr));
        if (!a->worker) throw 8;
        state = a;
        return 0;
    } catch (...) {
        if (a) { if (a->reader) FreeLibrary(a->reader); if (a->cuda) FreeLibrary(a->cuda); delete a; }
        return 8;
    }
}

static bool valid_context(State& a) noexcept {
    Context current = nullptr;
    return GetCurrentThreadId() == a.owner && !a.closing.load(std::memory_order_acquire) && !a.current(&current) && current == a.context;
}
API int mbx_pa_post(uint64_t stream, int count) noexcept {
    State* a = state;
    if (!a || count <= 0 || count > a->maximum || !valid_context(*a)) return 1;
    return a->write(reinterpret_cast<Stream>(stream), a->request_device, static_cast<unsigned>(count), 0) ? 1 : 0;
}
API int mbx_pa_wait(uint64_t stream) noexcept {
    State* a = state;
    if (!a || !valid_context(*a)) return 1;
    if (a->wait(reinterpret_cast<Stream>(stream), a->done_device, 1, 1)) return 2;
    return a->write(reinterpret_cast<Stream>(stream), a->done_device, 0, 0) ? 3 : 0;
}
API int mbx_pa_failed() noexcept { std::lock_guard guard(state_mutex); return state ? state->failed.load(std::memory_order_acquire) : 0; }
API int64_t mbx_pa_jobs() noexcept { std::lock_guard guard(state_mutex); return state ? state->jobs.load() : 0; }
API double mbx_pa_read_us() noexcept { std::lock_guard guard(state_mutex); return state ? state->read_us.load() : 0.0; }

// Extra Windows lifecycle API, called only AFTER every original Graph is reset
// and its actual replay stream is drained. It never fabricates flag completion.
API int mbx_pa_shutdown() noexcept {
    std::lock_guard guard(state_mutex);
    State* a = state; if (!a) return 0;
    if (GetCurrentThreadId() != a->owner || load(a->request) || load(a->done) || a->busy.load()) return 9;
    a->closing.store(true, std::memory_order_release);
    a->stop.store(true, std::memory_order_release);
    if (WaitForSingleObject(a->worker, 5000) != WAIT_OBJECT_0) return 10; // keep strong owner
    CloseHandle(a->worker);
    FreeLibrary(a->reader); if (a->cuda) FreeLibrary(a->cuda);
    state = nullptr; delete a; return 0;
}
