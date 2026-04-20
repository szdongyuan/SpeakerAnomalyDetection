from ui.custom_ui_widget.widgets import MessageBox


class PopupUtils(object):

    @staticmethod
    def save_popup(parent, success_flag=True):
        save_msg = MessageBox(parent)
        if success_flag:
            save_msg.setIcon(MessageBox.Information)
            save_msg.setText("设置成功")
            save_msg.setWindowTitle("设置成功")
        else:
            save_msg.setIcon(MessageBox.Critical)
            save_msg.setText("设置失败，请重试")
            save_msg.setWindowTitle("设置失败")
        save_msg.exec_()
