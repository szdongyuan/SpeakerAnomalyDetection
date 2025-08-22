import uuid


def get_mac_address():
    mac = uuid.getnode()
    mac_address = ":".join(("%012x" % mac)[i : i + 2] for i in range(0, 12, 2))
    return mac_address
