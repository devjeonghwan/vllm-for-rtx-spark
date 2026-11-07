"""A task-owned Win32 notification icon; no extra GUI package or detached server."""
import ctypes
from ctypes import wintypes as w
import os
from pathlib import Path
import threading
import webbrowser


class Tray:
    def __init__(self, url):
        self.url = url
        self.hwnd = None
        self.thread = None
        self.ready = threading.Event()
        self.error = None

    def start(self):
        if os.name != "nt":
            raise RuntimeError("Windows tray is unavailable")
        self.thread = threading.Thread(target=self._run, name="vllm-monitor-tray", daemon=True)
        self.thread.start()
        if not self.ready.wait(5):
            self.close()
            raise RuntimeError("Monitor tray startup timed out")
        if self.error is not None:
            raise RuntimeError("Monitor tray startup failed") from self.error

    def close(self):
        if self.hwnd:
            user = ctypes.WinDLL("user32", use_last_error=True)
            user.PostMessageW.argtypes = (w.HWND, w.UINT, w.WPARAM, w.LPARAM)
            user.PostMessageW.restype = w.BOOL
            user.PostMessageW(self.hwnd, 0x0010, 0, 0)  # WM_CLOSE
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(3)
            if self.thread.is_alive():
                raise RuntimeError("Monitor tray did not stop")

    def _run(self):
        user = ctypes.WinDLL("user32", use_last_error=True)
        shell = ctypes.WinDLL("shell32", use_last_error=True)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        result_type = ctypes.c_ssize_t  # LRESULT is pointer-sized on native ARM64.
        callback_type = ctypes.WINFUNCTYPE(result_type, w.HWND, w.UINT, w.WPARAM, w.LPARAM)

        class WndClass(ctypes.Structure):
            _fields_ = [("style", w.UINT), ("proc", callback_type), ("cls_extra", ctypes.c_int),
                        ("wnd_extra", ctypes.c_int), ("instance", w.HINSTANCE), ("icon", w.HICON),
                        ("cursor", w.HANDLE), ("background", w.HBRUSH), ("menu", w.LPCWSTR),
                        ("name", w.LPCWSTR)]

        class NotifyIcon(ctypes.Structure):
            _fields_ = [("size", w.DWORD), ("hwnd", w.HWND), ("id", w.UINT),
                        ("flags", w.UINT), ("callback", w.UINT), ("icon", w.HICON),
                        ("tip", w.WCHAR * 128), ("state", w.DWORD), ("state_mask", w.DWORD),
                        ("info", w.WCHAR * 256), ("version", w.UINT), ("title", w.WCHAR * 64),
                        ("info_flags", w.DWORD), ("guid", ctypes.c_byte * 16), ("balloon", w.HICON)]

        def api(dll, name, result, *arguments):
            fn = getattr(dll, name)
            fn.restype, fn.argtypes = result, arguments
            return fn

        module = api(kernel, "GetModuleHandleW", w.HINSTANCE, w.LPCWSTR)(None)
        register = api(user, "RegisterClassW", w.ATOM, ctypes.POINTER(WndClass))
        unregister = api(user, "UnregisterClassW", w.BOOL, w.LPCWSTR, w.HINSTANCE)
        create = api(user, "CreateWindowExW", w.HWND, w.DWORD, w.LPCWSTR, w.LPCWSTR, w.DWORD,
                     ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                     w.HWND, w.HMENU, w.HINSTANCE, w.LPVOID)
        default = api(user, "DefWindowProcW", result_type, w.HWND, w.UINT, w.WPARAM, w.LPARAM)
        destroy = api(user, "DestroyWindow", w.BOOL, w.HWND)
        notify = api(shell, "Shell_NotifyIconW", w.BOOL, w.DWORD, ctypes.POINTER(NotifyIcon))
        get_message = api(user, "GetMessageW", ctypes.c_int, ctypes.POINTER(w.MSG), w.HWND, w.UINT, w.UINT)
        translate = api(user, "TranslateMessage", w.BOOL, ctypes.POINTER(w.MSG))
        dispatch = api(user, "DispatchMessageW", result_type, ctypes.POINTER(w.MSG))
        post_quit = api(user, "PostQuitMessage", None, ctypes.c_int)
        load_image = api(user, "LoadImageW", w.HANDLE, w.HINSTANCE, w.LPCWSTR,
                         w.UINT, ctypes.c_int, ctypes.c_int, w.UINT)
        destroy_icon = api(user, "DestroyIcon", w.BOOL, w.HICON)
        make_menu = api(user, "CreatePopupMenu", w.HMENU)
        append_menu = api(user, "AppendMenuW", w.BOOL, w.HMENU, w.UINT, ctypes.c_size_t, w.LPCWSTR)
        track_menu = api(user, "TrackPopupMenu", w.UINT, w.HMENU, w.UINT, ctypes.c_int, ctypes.c_int,
                         ctypes.c_int, w.HWND, w.LPVOID)
        destroy_menu = api(user, "DestroyMenu", w.BOOL, w.HMENU)
        cursor = api(user, "GetCursorPos", w.BOOL, ctypes.POINTER(w.POINT))
        foreground = api(user, "SetForegroundWindow", w.BOOL, w.HWND)
        taskbar_created = api(user, "RegisterWindowMessageW", w.UINT, w.LPCWSTR)("TaskbarCreated")
        name = f"VllmMonitor-{os.getpid()}-{id(self):x}"
        data = NotifyIcon()
        installed = False
        registered = False

        def open_monitor():
            threading.Thread(target=webbrowser.open, args=(self.url,), daemon=True).start()

        @callback_type
        def procedure(hwnd, message, wp, lp):
            if message == 0x8001:  # WM_APP + 1: icon callback
                if lp in (0x0202, 0x0203):  # left click / double click
                    open_monitor()
                elif lp == 0x0205:  # right click
                    menu = make_menu()
                    try:
                        append_menu(menu, 0, 1, "Open monitor")
                        append_menu(menu, 0, 2, "Close tray icon")
                        point = w.POINT()
                        cursor(ctypes.byref(point))
                        foreground(hwnd)
                        choice = track_menu(menu, 0x0100 | 0x0002, point.x, point.y, 0, hwnd, None)
                        if choice == 1:
                            open_monitor()
                        elif choice == 2:
                            destroy(hwnd)
                    finally:
                        destroy_menu(menu)
                return 0
            if taskbar_created and message == taskbar_created:
                notify(0, ctypes.byref(data))  # Explorer restart: restore this icon only.
                return 0
            if message == 0x0010:  # WM_CLOSE
                destroy(hwnd)
                return 0
            if message == 0x0002:  # WM_DESTROY
                notify(2, ctypes.byref(data))
                self.hwnd = None
                post_quit(0)
                return 0
            return default(hwnd, message, wp, lp)

        try:
            wc = WndClass()
            wc.proc, wc.instance, wc.name = procedure, module, name
            if not register(ctypes.byref(wc)):
                raise ctypes.WinError(ctypes.get_last_error())
            registered = True
            self.hwnd = create(0, name, "vLLM Monitor", 0, 0, 0, 0, 0, None, None, module, None)
            if not self.hwnd:
                raise ctypes.WinError(ctypes.get_last_error())
            data.size, data.hwnd, data.id = ctypes.sizeof(data), self.hwnd, 1
            data.flags, data.callback = 1 | 2 | 4, 0x8001
            # LR_LOADFROMFILE | LR_DEFAULTSIZE, without LR_SHARED: this thread owns HICON.
            data.icon = load_image(None, str(Path(__file__).with_name("static") / "favicon.ico"),
                                   1, 0, 0, 0x0010 | 0x0040)
            if not data.icon:
                raise ctypes.WinError(ctypes.get_last_error())
            data.tip = "vLLM Monitor"
            if not notify(0, ctypes.byref(data)):
                raise ctypes.WinError(ctypes.get_last_error())
            installed = True
            self.ready.set()
            message = w.MSG()
            while True:
                status = get_message(ctypes.byref(message), None, 0, 0)
                if status == 0:
                    break
                if status < 0:
                    raise ctypes.WinError(ctypes.get_last_error())
                translate(ctypes.byref(message))
                dispatch(ctypes.byref(message))
        except BaseException as error:
            self.error = error
        finally:
            if installed:
                notify(2, ctypes.byref(data))
            if self.hwnd:
                destroy(self.hwnd)
                self.hwnd = None
            if registered:
                unregister(name, module)
            if data.icon:
                destroy_icon(data.icon)
                data.icon = None
            self.ready.set()
