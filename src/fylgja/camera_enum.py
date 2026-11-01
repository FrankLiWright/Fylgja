"""摄像头枚举（方案书 6 步骤 3）：设备标识 + 友好名，不依赖裸索引猜测设备。

DirectShow 类目枚举（comtypes 手写 COM 接口），索引顺序与 OpenCV CAP_DSHOW 一致：
  ICreateDevEnum → CreateClassEnumerator(VideoInputDeviceCategory) → IEnumMoniker
  → IMoniker: FriendlyName（IPropertyBag）+ DisplayName（PnP 稳定设备路径）。

配置存 ``path``（稳定）+ ``index``（回退），启动时重新枚举按 path 解析当前索引。
"""
from __future__ import annotations

from dataclasses import dataclass

from .errors import Attribution, CameraError

CLSID_SystemDeviceEnum = "{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}"
CLSID_VideoInputDeviceCategory = "{860BB310-5D01-11d0-BD3B-00A0C911CE86}"
IID_ICreateDevEnum = "{29840822-5B84-11D0-BD3B-00A0C911CE86}"
IID_IEnumMoniker = "{00000102-0000-0000-C000-000000000046}"
IID_IMoniker = "{0000000F-0000-0000-C000-000000000046}"
IID_IPropertyBag = "{55272A00-42CB-11CE-8135-00AA004BB851}"


@dataclass(frozen=True)
class CameraDevice:
    index: int      # DirectShow / OpenCV CAP_DSHOW 索引
    name: str       # 友好名
    path: str       # 稳定设备标识（moniker DisplayName，PnP 路径）
    backend: str = "dshow"


def _make_interfaces():
    """按调用点所需的 vtable 顺序定义 COM 接口（IMoniker 需完整序号占位）。"""
    import comtypes
    from comtypes import COMMETHOD, GUID, HRESULT, IUnknown
    from comtypes.automation import VARIANT
    from ctypes import POINTER, c_ulong, c_wchar_p, c_void_p, c_int

    class ICreateDevEnum(IUnknown):
        _iid_ = GUID(IID_ICreateDevEnum)
        _methods_ = [
            COMMETHOD([], HRESULT, 'CreateClassEnumerator',
                      (['in'], POINTER(GUID), 'clsidDeviceClass'),
                      (['out'], POINTER(POINTER(IUnknown)), 'ppEnumMoniker'),
                      (['in'], c_ulong, 'dwFlags')),
        ]

    class IEnumMoniker(IUnknown):
        _iid_ = GUID(IID_IEnumMoniker)
        _methods_ = [
            COMMETHOD([], HRESULT, 'Next',
                      (['in'], c_ulong, 'celt'),
                      (['out'], POINTER(POINTER(IUnknown)), 'rgelt'),
                      (['out'], POINTER(c_ulong), 'pceltFetched')),
        ]

    class IMoniker(IUnknown):
        _iid_ = GUID(IID_IMoniker)
        _methods_ = [
            COMMETHOD([], HRESULT, 'GetClassID', (['out'], POINTER(GUID), 'pClassID')),   # 3
            COMMETHOD([], HRESULT, 'IsDirty'),                                            # 4
            COMMETHOD([], HRESULT, 'Load', (['in'], c_void_p, 'pStm')),                   # 5
            COMMETHOD([], HRESULT, 'Save', (['in'], c_void_p, 'pStm'),
                      (['in'], c_int, 'fClearDirty')),                                    # 6
            COMMETHOD([], HRESULT, 'GetSizeMax', (['out'], POINTER(c_void_p), 'pcbSize')),  # 7
            COMMETHOD([], HRESULT, 'BindToObject', (['in'], c_void_p, 'pbc'),
                      (['in'], c_void_p, 'pmkToLeft'),
                      (['in'], POINTER(GUID), 'riidResult'),
                      (['out'], POINTER(c_void_p), 'ppvResult')),                          # 8
            COMMETHOD([], HRESULT, 'BindToStorage', (['in'], c_void_p, 'pbc'),
                      (['in'], c_void_p, 'pmkToLeft'),
                      (['in'], POINTER(GUID), 'riid'),
                      (['out'], POINTER(c_void_p), 'ppvObj')),                            # 9
            COMMETHOD([], HRESULT, 'Reduce', (['in'], c_void_p, 'pbc'),
                      (['in'], c_ulong, 'dwReduceHowFar'),
                      (['in'], c_void_p, 'ppmkToLeft'),
                      (['out'], POINTER(c_void_p), 'ppmkReduced')),                       # 10
            COMMETHOD([], HRESULT, 'ComposeWith', (['in'], c_void_p, 'pmkFirst'),
                      (['in'], c_int, 'fOnlyIfNotGeneric'),
                      (['out'], POINTER(c_void_p), 'ppmkComposite')),                     # 11
            COMMETHOD([], HRESULT, 'Enum', (['in'], c_int, 'fForward'),
                      (['out'], POINTER(c_void_p), 'ppenumMoniker')),                     # 12
            COMMETHOD([], HRESULT, 'IsEqual', (['in'], c_void_p, 'pmkOtherMoniker')),     # 13
            COMMETHOD([], HRESULT, 'Hash', (['out'], POINTER(c_ulong), 'pdwHash')),       # 14
            COMMETHOD([], HRESULT, 'IsRunning', (['in'], c_void_p, 'pbc'),
                      (['in'], c_void_p, 'pmkToLeft'),
                      (['in'], c_void_p, 'pmkNewlyRunning')),                             # 15
            COMMETHOD([], HRESULT, 'GetTimeOfLastChange', (['in'], c_void_p, 'pbc'),
                      (['in'], c_void_p, 'pmkToLeft'),
                      (['out'], POINTER(c_void_p), 'pFileTime')),                         # 16
            COMMETHOD([], HRESULT, 'Inverse', (['out'], POINTER(c_void_p), 'ppmk')),      # 17
            COMMETHOD([], HRESULT, 'CommonPrefixWith', (['in'], c_void_p, 'pmkOther'),
                      (['out'], POINTER(c_void_p), 'ppmkPrefix')),                        # 18
            COMMETHOD([], HRESULT, 'RelativePathTo', (['in'], c_void_p, 'pmkOther'),
                      (['out'], POINTER(c_void_p), 'ppmkRelPath')),                       # 19
            COMMETHOD([], HRESULT, 'GetDisplayName', (['in'], c_void_p, 'pbc'),
                      (['in'], c_void_p, 'pmkToLeft'),
                      (['out'], POINTER(c_wchar_p), 'ppszDisplayName')),                  # 20
        ]

    class IPropertyBag(IUnknown):
        _iid_ = GUID(IID_IPropertyBag)
        _methods_ = [
            COMMETHOD([], HRESULT, 'Read',
                      (['in'], c_wchar_p, 'pszPropName'),
                      (['out'], POINTER(VARIANT), 'pVar'),
                      (['in'], c_void_p, 'pErrorLog')),
        ]

    return comtypes, ICreateDevEnum, IEnumMoniker, IMoniker, IPropertyBag


def enumerate_dshow_devices() -> list[CameraDevice]:
    """DirectShow 类目枚举；索引顺序与 CAP_DSHOW 一致。空列表 = 无设备。"""
    try:
        comtypes, ICreateDevEnum, IEnumMoniker, IMoniker, IPropertyBag = _make_interfaces()
    except ImportError as e:
        raise CameraError(
            f"comtypes 不可用，无法枚举摄像头（{e}）",
            Attribution.SYSTEM,
        ) from e

    from ctypes import POINTER, c_void_p, cast as ccast, c_ulong
    from comtypes import GUID, IUnknown

    devices: list[CameraDevice] = []
    comtypes.CoInitialize()
    try:
        dev_enum = comtypes.CoCreateInstance(
            GUID(CLSID_SystemDeviceEnum), ICreateDevEnum,
        )
        enum = dev_enum.CreateClassEnumerator(GUID(CLSID_VideoInputDeviceCategory), 0)
        if not enum:
            return []  # 无视频输入设备
        enum = enum.QueryInterface(IEnumMoniker)
        index = 0
        while True:
            batch = enum.Next(1)
            if not batch or not batch[0]:
                break
            mk = batch[0].QueryInterface(IMoniker)
            raw = mk.BindToStorage(None, None, GUID(IID_IPropertyBag))
            bag = ccast(c_void_p(raw), POINTER(IUnknown)).QueryInterface(IPropertyBag)
            name = str(bag.Read("FriendlyName", None))
            path = str(mk.GetDisplayName(None, None))
            devices.append(CameraDevice(index=index, name=name, path=path))
            index += 1
    finally:
        comtypes.CoUninitialize()
    return devices


def resolve_camera(devices: list[CameraDevice], *, path: str = "",
                   index: int = 0) -> CameraDevice:
    """按稳定路径优先解析；路径丢失/为空时回退索引（方案书 6：不得盲猜设备）。"""
    if not devices:
        raise CameraError(
            "枚举不到任何摄像头：检查 Windows 相机隐私开关"
            "（设置→隐私和安全性→相机→允许桌面应用访问）或设备被占用（方案书 8.3）"
        )
    if path:
        for dev in devices:
            if dev.path == path:
                return dev
        # 配置的设备已不在：显式报错，不静默替换（方案书 6 禁止静默回退）
        names = "; ".join(f"{d.index}:{d.name}" for d in devices)
        raise CameraError(
            f"配置的摄像头设备已丢失（path={path!r}）。当前可用：{names}。请重新选择。"
        )
    for dev in devices:
        if dev.index == index:
            return dev
    names = "; ".join(f"{d.index}:{d.name}" for d in devices)
    raise CameraError(f"摄像头索引 {index} 不存在。当前可用：{names}")
