import pywinusb.hid as hid

from base.log_manager import LogManager


class BarcodeScanner(object):

    def __init__(self):
        super().__init__()
        self.received_data = False
        self.barcode = None
        self.scanning = False
        self.default_logger = LogManager.set_log_handler("core")

    @staticmethod
    def find_scanner(vendor_id, product_id):
        all_hid = hid.find_all_hid_devices()
        for device in all_hid:
            if device.vendor_id == vendor_id and device.product_id == product_id:
                return device
        return None

    def handle_raw_data(self, data):
        data_str = ''.join(chr(byte) for byte in data if 32 <= byte <= 126)
        start_index = data_str.find('"data":"')
        if start_index != -1:
            start_index += len('"data":"')
            end_index = data_str.find('"', start_index)
            if end_index != -1:
                barcode = data_str[start_index:end_index]
                return barcode
            else:
                self.default_logger.warning("End of barcode not found.")
        else:
            self.default_logger.warning("Start of barcode not found.")

    def raw_data_handler(self, data):
        barcode = self.handle_raw_data(data)
        if barcode:
            self.received_data = True
            self.barcode = barcode

    def read_raw_data(self, scanner):
        if scanner:
            try:
                scanner.open()
                scanner.set_raw_data_handler(self.raw_data_handler)
                self.scanning = True
                while scanner.is_plugged() and not self.received_data and self.scanning:
                    pass
                return self.barcode
            except Exception as e:
                self.default_logger.error(f"Error reading data:{e}")
            finally:
                if scanner:
                    scanner.close()
                    self.received_data = False
                    self.scanning = False
                    self.barcode = None

    def stop_scanning(self):
        self.scanning = False



if __name__ == "__main__":
    reader = BarcodeScanner()
    scanner = reader.find_scanner(vendor_id=0x2f50, product_id=0x300)
    barcode = reader.read_raw_data(scanner)
