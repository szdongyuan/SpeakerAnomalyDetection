from PyQt5.QtWidgets import QMessageBox


class PopupUtils(object):

    @staticmethod
    def save_popup(parent, success_flag=True):
        save_msg = QMessageBox(parent)
        if success_flag:
            save_msg.setIcon(QMessageBox.Information)
            save_msg.setText("设置成功")
            save_msg.setWindowTitle("设置成功")
        else:
            save_msg.setIcon(QMessageBox.Critical)
            save_msg.setText("设置失败，请重试")
            save_msg.setWindowTitle("设置失败")
        save_msg.exec_()


def check_upper_lower_limit(config_data: dict, parent):
    if config_data["limit_checked"] is False:
        return False

    if int(config_data["upper_limit"]) <= int(config_data["lower_limit"]):
        QMessageBox.warning(parent, "设置警告", "上下限配置数据错误，请检查配置!")
        return True
    else:
        return False
