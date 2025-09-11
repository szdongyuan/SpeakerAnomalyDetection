import ctypes
from ctypes import wintypes
import socket

# 定义 Windows API 所需的常量和结构体
# 从 iphlpapi.h
MAX_ADAPTER_ADDRESS_LENGTH = 8
GAA_FLAG_INCLUDE_PREFIX = 0x0010  # GetAdaptersAddresses flags

# AF_INET, AF_INET6, AF_UNSPEC
AF_UNSPEC = socket.AF_UNSPEC


# 定义 SOCKET_ADDRESS 结构体
class SOCKADDR(ctypes.Structure):
    _fields_ = [("sa_family", wintypes.USHORT), ("sa_data", ctypes.c_char * 14)]


class SOCKET_ADDRESS(ctypes.Structure):
    _fields_ = [("lpSockaddr", ctypes.POINTER(SOCKADDR)), ("iSockaddrLength", ctypes.c_int)]


# 定义 IP_ADAPTER_ADDRESSES 结构体
class IP_ADAPTER_ADDRESSES(ctypes.Structure):
    pass


# 结构体中包含指向自身的指针，用于链表结构
IP_ADAPTER_ADDRESSES._fields_ = [
    ("Length", wintypes.ULONG),
    ("IfIndex", wintypes.DWORD),
    ("Next", ctypes.POINTER(IP_ADAPTER_ADDRESSES)),
    ("AdapterName", ctypes.c_char_p),
    ("FirstUnicastAddress", ctypes.c_void_p),  # Placeholder
    ("FirstAnycastAddress", ctypes.c_void_p),  # Placeholder
    ("FirstMulticastAddress", ctypes.c_void_p),  # Placeholder
    ("FirstDnsServerAddress", ctypes.c_void_p),  # Placeholder
    ("DnsSuffix", ctypes.c_wchar_p),
    ("Description", ctypes.c_wchar_p),
    ("FriendlyName", ctypes.c_wchar_p),
    ("PhysicalAddress", wintypes.BYTE * MAX_ADAPTER_ADDRESS_LENGTH),
    ("PhysicalAddressLength", wintypes.DWORD),
    ("Flags", wintypes.DWORD),
    ("Mtu", wintypes.DWORD),
    ("IfType", wintypes.DWORD),
    ("OperStatus", ctypes.c_int),
    # ... 其他字段可以根据需要添加
]

# IANA 定义的接口类型常量
IF_TYPE_ETHERNET_CSMACD = 6
IF_TYPE_IEEE80211 = 71


# 用于排除虚拟和不需要的适配器的关键字列表
EXCLUDE_KEYWORDS = [
    "virtual",
    "hyper-v",
    "vmware",
    "virtualbox",
    "loopback",
    "teredo",
    "isatap",
    "bluetooth",
    "tap-windows",  # Common for VPNs
    "Microsoft Kernel",
]

DESCRIPTION_KEYWORDS = [
    "realtek",
    "intel",
    "broadcom",
    "amd",
    "ethernet",
    "gigabit",
    "lan",
    "network",
    "integrated",
    "pci",
    "pcie",
    "pci-e",
]


def get_mac_address():
    """通过Windows系统接口 GetAdaptersAddresses 获取所有网络适配器MAC地址"""
    physical_adapters = []

    # 加载IP Helper库
    iphlpapi = ctypes.WinDLL("iphlpapi.dll")

    # 第一次调用获取所需缓冲区大小
    buffer_size = wintypes.ULONG(0)
    # AF_UNSPEC 获取 IPv4 和 IPv6
    # GAA_FLAG_INCLUDE_PREFIX 是必需的
    result = iphlpapi.GetAdaptersAddresses(AF_UNSPEC, GAA_FLAG_INCLUDE_PREFIX, None, None, ctypes.byref(buffer_size))

    # 如果第一次调用成功或返回的不是缓冲区大小错误，则说明有问题
    if result != 111:  # ERROR_BUFFER_OVERFLOW
        raise ctypes.WinError(result)

    # 分配缓冲区
    buffer = ctypes.create_string_buffer(buffer_size.value)
    adapter_addresses = ctypes.cast(buffer, ctypes.POINTER(IP_ADAPTER_ADDRESSES))

    # 第二次调用获取实际的适配器信息
    result = iphlpapi.GetAdaptersAddresses(
        AF_UNSPEC, GAA_FLAG_INCLUDE_PREFIX, None, adapter_addresses, ctypes.byref(buffer_size)
    )
    if result != 0:  # NO_ERROR
        raise ctypes.WinError(result)

    # 遍历所有适配器
    adapter = adapter_addresses
    while adapter:
        adapter_info = adapter.contents

        # 过滤条件 1: 必须有有效的MAC地址
        if not adapter_info.PhysicalAddressLength > 0:
            adapter = adapter_info.Next
            continue

        # 过滤条件 2: 必须是物理网卡类型 (以太网或Wi-Fi)
        if adapter_info.IfType not in [IF_TYPE_ETHERNET_CSMACD, IF_TYPE_IEEE80211]:
            adapter = adapter_info.Next
            continue

        # 过滤条件 3: 描述和名称中不能包含排除关键字
        desc_lower = adapter_info.Description.lower()
        name_lower = adapter_info.FriendlyName.lower()
        is_excluded = any(keyword in desc_lower or keyword in name_lower for keyword in EXCLUDE_KEYWORDS)
        is_description = any(keyword in desc_lower for keyword in DESCRIPTION_KEYWORDS)

        if not is_excluded and is_description:
            mac = ":".join(f"{b:02X}" for b in adapter_info.PhysicalAddress[: adapter_info.PhysicalAddressLength])
            physical_adapters.append(
                {
                    "name": adapter_info.FriendlyName,
                    "description": adapter_info.Description,
                    "mac_address": mac,
                    "type": adapter_info.IfType,
                    "status": adapter_info.OperStatus,
                }
            )

        adapter = adapter_info.Next
    if len(physical_adapters) > 0:
        address = physical_adapters[0]["mac_address"].lower()
        return address

    raise Exception("No MAC address found")
