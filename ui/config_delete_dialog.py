"""Select registered configurations for deletion in one scrollable checklist."""

from dataclasses import replace

from PyQt5.QtCore import QRect, QSize, Qt
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPalette
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton, QStyle,
    QStyledItemDelegate, QStyleOptionViewItem, QVBoxLayout,
)

from base.config_deletion import ConfigDeletionError
from ui.config_dialog_base import ConfigDialogBase


STATUS_ROLE = Qt.UserRole + 2


class DeletionItemDelegate(QStyledItemDelegate):
    """Keep names aligned while showing availability separately from the name."""

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        return QSize(size.width(), max(size.height(), option.fontMetrics.height() + 8))

    def paint(self, painter, option, index):
        view = QStyleOptionViewItem(option)
        self.initStyleOption(view, index)
        view.state &= ~QStyle.State_HasFocus
        style = view.widget.style()
        painter.save()
        style.drawPrimitive(QStyle.PE_PanelItemViewItem, view, painter, view.widget)

        check = QStyleOptionViewItem(view)
        check.rect = style.subElementRect(QStyle.SE_ItemViewItemCheckIndicator, view, view.widget)
        check.state &= ~(QStyle.State_On | QStyle.State_Off | QStyle.State_NoChange)
        check.state |= QStyle.State_On if view.checkState == Qt.Checked else QStyle.State_Off
        if not index.flags() & Qt.ItemIsUserCheckable:
            check.state &= ~QStyle.State_Enabled
        style.drawPrimitive(QStyle.PE_IndicatorItemViewItemCheck, check, painter, view.widget)

        text_rect = style.subElementRect(QStyle.SE_ItemViewItemText, view, view.widget)
        status = index.data(STATUS_ROLE)
        if status:
            tag_font = QFont(view.font)
            tag_font.setPixelSize(13)
            tag_metrics = QFontMetrics(tag_font)
            tag = QRect(0, 0, tag_metrics.horizontalAdvance(status) + 16, tag_metrics.height() + 6)
            tag.moveRight(view.rect.right() - 8)
            tag.moveTop(view.rect.center().y() - tag.height() // 2)
            text_rect.setRight(tag.left() - 10)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#edf1f5"))
            painter.drawRoundedRect(tag, 4, 4)
            painter.setFont(tag_font)
            painter.setPen(QColor("#68788a"))
            painter.drawText(tag, Qt.AlignCenter, status)
        painter.setFont(view.font)
        painter.setPen(view.palette.color(QPalette.Text))
        text = view.fontMetrics.elidedText(view.text, Qt.ElideMiddle, max(0, text_rect.width()))
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, text)
        painter.restore()


class ConfigDeleteDialog(ConfigDialogBase):
    def __init__(self, service, kind, parent=None, *, drafts=None, busy=None,
                 unsaved=None, completed=None, failed=None, current_product_file=None):
        super().__init__(parent)
        self.service = service
        self.kind = kind
        self.drafts = drafts or (lambda: ())
        self.busy = busy or (lambda: False)
        self.unsaved = unsaved or (lambda target: False)
        self.completed = completed
        self.failed = failed
        self.current_product_file = current_product_file
        self.completed_targets = []
        self._executing = False
        self._blocked = False
        self.setWindowTitle("删除产品测试配置" if kind == "product" else "删除测试队列配置")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setModal(True)
        self.resize(500, 360)
        self.setMinimumSize(460, 300)
        self.config_list = QListWidget(self)
        self.config_list.setObjectName("deletionList")
        self.config_list.setItemDelegate(DeletionItemDelegate(self.config_list))
        self.config_list.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.config_list.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.config_list.setTextElideMode(Qt.ElideMiddle)
        self.config_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.details = QPlainTextEdit(self)
        self.details.setObjectName("deletionDetails")
        self.details.setReadOnly(True)
        self.details.setFrameShape(QPlainTextEdit.NoFrame)
        self.details.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.details.document().setDocumentMargin(0)
        self.details.viewport().setAutoFillBackground(False)
        self.apply_config_dialog_theme("""
            QListWidget#deletionList {
                font-size: 18px; border: 1px solid #cbd5e1; border-radius: 4px; padding: 4px;
                selection-background-color: #e7f0fa;
            }
            QPlainTextEdit#deletionDetails {
                border: none; background: transparent; padding: 0; font-size: 16px;
            }
        """)
        self.config_list.ensurePolished()
        details_font = self.config_list.font()
        details_font.setPixelSize(16)
        self.details.setFont(details_font)
        self.details.ensurePolished()
        self.details.horizontalScrollBar().rangeChanged.connect(self._size_details)
        self.details.textChanged.connect(self._size_details)
        self._size_details()
        self.selection_count = QLabel("已选 0 项", self)
        self.delete_button = QPushButton("删除", self)
        self.delete_button.setAutoDefault(False)
        self.cancel_button = QPushButton("取消", self)
        self.cancel_button.setDefault(True)
        self.cancel_button.clicked.connect(self.reject)
        self.delete_button.clicked.connect(self._execute)
        self.config_list.itemChanged.connect(self._selection_changed)
        self.config_list.currentItemChanged.connect(self._selection_changed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        layout.addWidget(QLabel("勾选要删除的配置：", self))
        layout.addWidget(self.config_list, 1)
        layout.addWidget(self.details)
        layout.addSpacing(8)
        buttons = QHBoxLayout()
        buttons.addWidget(self.selection_count)
        buttons.addStretch()
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.delete_button)
        layout.addLayout(buttons)
        self._load_targets()

    def checked_targets(self):
        return [self.config_list.item(index).data(Qt.UserRole)
                for index in range(self.config_list.count())
                if self.config_list.item(index).flags() & Qt.ItemIsUserCheckable
                and self.config_list.item(index).checkState() == Qt.Checked]

    def _size_details(self):
        # Show up to four complete rows without leaving a large empty message area.
        rows = min(4, max(1, self.details.document().blockCount()))
        height = self.details.fontMetrics().lineSpacing() * rows + 1
        if self.details.horizontalScrollBar().maximum() > 0:
            height += self.details.style().pixelMetric(QStyle.PM_ScrollBarExtent)
        self.details.setFixedHeight(height)

    @staticmethod
    def _error_message(target, error):
        if error.references:
            return "被以下工况引用，暂不能删除：\n" + "\n".join(error.references)
        return f"“{target.names[0]}”：{error}"

    def _load_targets(self, message=""):
        self.config_list.blockSignals(True)
        self.config_list.clear()
        try:
            targets = self.service.list_targets(self.kind)
            for target in targets:
                name = " / ".join(target.names)
                item = QListWidgetItem(name, self.config_list)
                item.setData(Qt.UserRole, target)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                item.setToolTip(" / ".join(target.names) + "\n" + target.path)
                if self.kind == "queue":
                    try:
                        self.service.check(target, drafts=self.drafts())
                    except ConfigDeletionError as error:
                        reason = "已引用" if error.references else "不可删除"
                        item.setData(STATUS_ROLE, reason)
                        item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
                        detail = self._error_message(target, error)
                        item.setData(Qt.UserRole + 1, detail)
                        item.setToolTip(item.toolTip() + "\n" + detail)
            if not targets and not message:
                message = "暂无可删除配置"
        except ConfigDeletionError as error:
            message = "\n".join(filter(None, (message, str(error))))
        finally:
            self.config_list.blockSignals(False)
        self.selection_count.setText("已选 0 项")
        self.details.setPlainText(message)
        self.delete_button.setEnabled(False)
        self.delete_button.setText("删除")

    def _check(self, target):
        if self.busy():
            raise ConfigDeletionError("当前轮次未结束、录音/分析尚未完成或配置状态异常，暂时不能删除。")
        self.service.check(target, drafts=self.drafts())

    def _selection_changed(self, *_items):
        self.delete_button.setEnabled(False)
        targets = self.checked_targets()
        self.selection_count.setText(f"已选 {len(targets)} 项")
        file_count = sum(target.remove_file for target in targets)
        action = "删除" if file_count or not targets else "移除"
        self.delete_button.setText(action)
        if not targets or self._blocked:
            self._show_details("")
            return
        missing_count = sum(target.file_stamp is None for target in targets)
        retained_count = len(targets) - file_count - missing_count
        actions = []
        if file_count:
            preserved = "测试队列和测试数据" if self.kind == "product" else "录音和测试结果"
            actions.append(f"删除 {file_count} 个配置文件，保留{preserved}。")
        if missing_count:
            actions.append(f"清理 {missing_count} 条失效记录（配置文件不存在）。")
        if retained_count:
            actions.append(f"移除 {retained_count} 条列表记录，保留原文件。")
        message = "\n".join(actions)
        has_unsaved_changes = any(self.unsaved(target) for target in targets)
        if self.kind == "product" and any(
            target.key == self.current_product_file for target in targets
        ):
            message += (
                "\n删除当前配置将丢弃未保存的修改，并返回主界面。"
                if has_unsaved_changes else "\n删除当前配置将返回主界面。"
            )
        elif has_unsaved_changes:
            message += "\n未保存的修改将丢弃。"
        try:
            for target in targets:
                self._check(target)
        except ConfigDeletionError as error:
            self.details.setPlainText(self._error_message(target, error))
            return
        self.delete_button.setEnabled(True)
        self._show_details(message)

    def _show_details(self, message):
        current = self.config_list.currentItem()
        reason = current.data(Qt.UserRole + 1) if current else None
        self.details.setPlainText(message or reason or "")

    def _execute(self):
        targets = self.checked_targets()
        if not targets or self._executing or self._blocked:
            return
        self._executing = True
        self.delete_button.setEnabled(False)
        self.config_list.setEnabled(False)
        try:
            # Validate the entire selection before writing the first item.
            for target in targets:
                self._check(target)
            registry_stamp = targets[0].registry_stamp
            for original in targets:
                target = replace(original, registry_stamp=registry_stamp)
                self._check(target)
                registry_stamp = self.service.delete(target, drafts=self.drafts())
                self.completed_targets.append(target)
                try:
                    if self.completed:
                        self.completed(target)
                except Exception as error:
                    # Persistence succeeded; stop so the user cannot repeat it.
                    raise ConfigDeletionError(
                        f"配置已{target.action}，界面刷新失败：{error}\n请检查后重启软件。",
                        inconsistent=True,
                    ) from error
        except ConfigDeletionError as error:
            message = self._error_message(target, error)
            if self.completed_targets:
                message = f"已完成 {len(self.completed_targets)} 项，其余未完成。\n" + message
            if error.inconsistent:
                self._blocked = True
                if self.failed:
                    self.failed(message)
                self.details.setPlainText(message)
            else:
                self._load_targets(message)
        else:
            self.accept()
        finally:
            self._executing = False
            self.config_list.setEnabled(not self._blocked)
            if self._blocked:
                self.cancel_button.setText("关闭")
