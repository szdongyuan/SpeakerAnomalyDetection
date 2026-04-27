import time

from PyQt5.QtCore import QEvent, QObject, Qt, QSignalBlocker
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QComboBox,
    QLineEdit,
    QPlainTextEdit,
    QTextEdit,
)


class BarcodeRouter(QObject):
    """
    把 SequenceWidget 里与“扫码枪（HID + 键盘楔入）”相关的逻辑集中到一个地方，便于维护。

    设计说明：
    - ctx 传入 SequenceWidget 实例（作为上下文），router 只负责“路由/拦截/缓冲/防抖提交”。
    - 条码最终提交仍走 ctx._commit_barcode(...)，保证业务流程不变（写回 S/N、复用统一校验/去重逻辑）。
    """

    def __init__(self, ctx):
        super().__init__(ctx)
        self.ctx = ctx

    @staticmethod
    def _is_input_widget(fw) -> bool:
        """
        输入控件白名单：
        - 任何输入控件获得焦点时，都不做“全局扫码捕获/吞键”，避免影响手动输入。
        - QComboBox（可编辑/不可编辑）都算白名单：不可编辑时也支持键盘选项跳转/回车确认等交互，
          不应被全局捕获逻辑吞掉。
        """
        if fw is None:
            return False
        if isinstance(fw, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)):
            return True
        if isinstance(fw, QComboBox):
            return True
        return False

    # -----------------------------
    # 复用/搬迁：normalize & should_commit
    # -----------------------------
    def normalize_barcode(self, text: str) -> str:
        if text is None:
            return ""
        return str(text).strip()

    @staticmethod
    def _fold_duplicated_payload(text: str) -> str:
        """
        Collapse immediate duplicated payload (e.g. "ABCABC" -> "ABC").
        Used to guard against scanners/devices injecting the same sequence twice
        into the line edit during one fast scan burst.
        """
        if not text or len(text) < 2 or (len(text) % 2) != 0:
            return text
        half = len(text) // 2
        left = text[:half]
        right = text[half:]
        return left if left == right else text

    def should_auto_commit_barcode(self, text: str, first_ts: float, last_ts: float) -> bool:
        """判断一段输入是否更像“扫码枪快速输入”，用于防抖自动提交。"""
        text = self.normalize_barcode(text)
        if not text:
            return False
        if len(text) < self.ctx._barcode_min_length_for_auto_commit:
            return False
        duration = max(0.0, last_ts - first_ts)
        return duration <= self.ctx._barcode_fast_input_max_seconds

    # -----------------------------
    # 复用/搬迁：S/N 输入框的 Enter / textChanged 防抖提交
    # -----------------------------
    def on_barcode_return_pressed(self):
        """键盘楔入模式：扫码枪通常会发送 Enter，触发此信号"""
        ctx = self.ctx
        if not ctx.barcode_scanner_box.isChecked():
            return
        ctx._barcode_debounce_timer.stop()
        ctx._commit_barcode(ctx.lineedit_s_or_n.text(), source="wedge_enter")

    def on_barcode_text_changed(self, _text: str):
        """
        键盘楔入模式：部分扫码枪不会发送 Enter，只会快速“敲”一串字符。
        用防抖：输入停止一小段时间后认为扫码结束。
        """
        ctx = self.ctx
        if not ctx.barcode_scanner_box.isChecked():
            return
        if not ctx.lineedit_s_or_n.isEnabled():
            return

        # Guard: some scanner paths may inject duplicated payload into the same
        # edit stream (e.g. "ABCABC" for one physical scan). Fold it early.
        # Only handle in scanner mode + very fast input path (this method),
        # so manual editing behavior is not affected.
        normalized = self.normalize_barcode(_text)
        folded = self._fold_duplicated_payload(normalized)
        if folded != normalized:
            try:
                with QSignalBlocker(ctx.lineedit_s_or_n):
                    ctx.lineedit_s_or_n.setText(folded)
            except Exception:
                ctx.lineedit_s_or_n.setText(folded)

        now = time.monotonic()
        if ctx._barcode_first_char_ts is None:
            ctx._barcode_first_char_ts = now
        ctx._barcode_last_char_ts = now
        ctx._barcode_debounce_timer.start()

    def on_barcode_debounce_timeout(self):
        """输入停顿后自动提交（无 Enter 扫码枪）"""
        ctx = self.ctx
        if not ctx.barcode_scanner_box.isChecked():
            return
        # 1) 优先处理“非 S/N 输入框焦点”下的全局捕获 buffer
        if (
            ctx._barcode_capture_buffer
            and ctx._barcode_capture_first_ts is not None
            and ctx._barcode_capture_last_ts is not None
        ):
            text = self.normalize_barcode(ctx._barcode_capture_buffer)
            if self.should_auto_commit_barcode(text, ctx._barcode_capture_first_ts, ctx._barcode_capture_last_ts):
                # 如果之前焦点在其它输入框，让字符先进入了那个输入框，这里把它恢复，避免污染
                try:
                    le = ctx._barcode_capture_target_lineedit
                    if le is not None and ctx._barcode_capture_target_text is not None:
                        with QSignalBlocker(le):
                            le.setText(ctx._barcode_capture_target_text)
                        if ctx._barcode_capture_target_cursor_pos is not None:
                            le.setCursorPosition(ctx._barcode_capture_target_cursor_pos)
                except Exception:
                    pass
                ctx._commit_barcode(text, source="wedge_global_debounce")
            else:
                # 不符合扫码特征则丢弃，避免误触发
                ctx._barcode_capture_buffer = ""
                ctx._barcode_capture_first_ts = None
                ctx._barcode_capture_last_ts = None
                ctx._barcode_capture_target_lineedit = None
                ctx._barcode_capture_target_text = None
                ctx._barcode_capture_target_cursor_pos = None
            return

        # 2) 处理 S/N 输入框自身的 textChanged 防抖
        text = self.normalize_barcode(ctx.lineedit_s_or_n.text())
        if not text:
            ctx._barcode_first_char_ts = None
            ctx._barcode_last_char_ts = None
            return
        if ctx._barcode_first_char_ts is None or ctx._barcode_last_char_ts is None:
            return
        if self.should_auto_commit_barcode(text, ctx._barcode_first_char_ts, ctx._barcode_last_char_ts):
            ctx._commit_barcode(text, source="wedge_debounce")
        else:
            # The text in S/N didn't look like a fast scan (too short, too
            # slow, or partial manual typing). Clear the per-keystroke
            # timestamps so the *next* burst is judged on its own merit;
            # otherwise ``first_char_ts`` keeps growing older forever and
            # ``should_auto_commit_barcode`` can never return True again
            # until the user manually empties S/N.
            ctx._barcode_first_char_ts = None
            ctx._barcode_last_char_ts = None

    # -----------------------------
    # 复用/搬迁：eventFilter 主逻辑（仅处理 KeyPress）
    # 返回值：
    # - True: 吞掉事件
    # - False: 不处理，让 Qt 继续派发
    # - None: 不是扫码逻辑的处理范围
    # -----------------------------
    def handle_keypress(self, obj, event):
        ctx = self.ctx
        if event.type() != QEvent.KeyPress:
            return None
        if not ctx.barcode_scanner_box.isChecked():
            return None

        now = time.monotonic()
        fw = QApplication.focusWidget()

        # HID 模式抑制窗口：在此之前忽略键盘输入（避免 HID + 键盘模式重复）
        ch = event.text()
        if ch and ch.isprintable() and now < ctx._hid_mode_active_until:
            return True

        # S/N 输入框：不在 router 内做复杂处理（保留给 SequenceWidget 自己的清空逻辑/默认输入）
        if fw is ctx.lineedit_s_or_n:
            return None

        # “最简焦点方案”：
        # 型号/计数输入框一律不拦截、不收集、不提交（保证手动输入 100% 正常）
        # 注意：键盘模式扫码会进入这些输入框，但不会触发条码提交/流程
        if fw is ctx.lineedit_type or fw is ctx.lineedit_count:
            return None

        # 扩展白名单：其它输入控件（SpinBox/可编辑ComboBox/文本框等）也不做全局捕获
        if self._is_input_widget(fw):
            return None

        key = event.key()

        # Enter：用于“其他位置全局捕获 buffer”立即提交
        if key in (Qt.Key_Return, Qt.Key_Enter):
            ctx._barcode_debounce_timer.stop()
            if (
                ctx._barcode_capture_buffer
                and ctx._barcode_capture_first_ts is not None
                and ctx._barcode_capture_last_ts is not None
            ):
                text = self.normalize_barcode(ctx._barcode_capture_buffer)
                if self.should_auto_commit_barcode(text, ctx._barcode_capture_first_ts, ctx._barcode_capture_last_ts):
                    # 恢复被输入框接收到的内容（若有）
                    try:
                        le = ctx._barcode_capture_target_lineedit
                        if le is not None and ctx._barcode_capture_target_text is not None:
                            with QSignalBlocker(le):
                                le.setText(ctx._barcode_capture_target_text)
                            if ctx._barcode_capture_target_cursor_pos is not None:
                                le.setCursorPosition(ctx._barcode_capture_target_cursor_pos)
                    except Exception:
                        pass
                    ctx._commit_barcode(text, source="wedge_global_enter")
                    return True  # 吞掉回车，避免触发按钮默认行为/输入框提交
            # 不像扫码则不拦截 Enter
            return None

        # 可打印字符
        if ch and ch.isprintable() and not ch.isspace():
            # 其他位置：全局捕获 buffer（键盘楔入）
            is_fast_input = (
                ctx._barcode_capture_last_ts is not None and (now - ctx._barcode_capture_last_ts) < 0.05
            )

            if ctx._barcode_capture_buffer and is_fast_input:
                ctx._barcode_capture_last_ts = now
                ctx._barcode_capture_buffer += ch
                ctx._barcode_debounce_timer.start()
                return True

            # Either the buffer is empty (truly first char) or the previous
            # capture went stale (gap > 50 ms). Both cases are semantically
            # "start a new capture from this character" -- the elif branch
            # used to forget to refresh the target lineedit/text/cursor and
            # would later restore the WRONG text box. Funnel through one
            # helper so the two paths can never drift apart again.
            self._begin_global_capture(ctx, ch, now, fw)
            return True

        return None

    def _begin_global_capture(self, ctx, ch, now, fw):
        """Start a fresh global capture window with ``ch`` as the first byte.

        Always re-snapshots the focus widget's text/cursor so a later
        debounce restore writes back the *currently* focused lineedit
        rather than whichever one was first focused several scans ago.
        Any non-QLineEdit focus clears the snapshot so we don't try to
        restore something we never captured.
        """
        ctx._barcode_debounce_timer.stop()
        ctx._barcode_capture_first_ts = now
        ctx._barcode_capture_last_ts = now
        ctx._barcode_capture_buffer = ch

        try:
            if isinstance(fw, QLineEdit):
                ctx._barcode_capture_target_lineedit = fw
                ctx._barcode_capture_target_text = fw.text()
                ctx._barcode_capture_target_cursor_pos = fw.cursorPosition()
            else:
                ctx._barcode_capture_target_lineedit = None
                ctx._barcode_capture_target_text = None
                ctx._barcode_capture_target_cursor_pos = None
        except Exception:
            ctx._barcode_capture_target_lineedit = None
            ctx._barcode_capture_target_text = None
            ctx._barcode_capture_target_cursor_pos = None

        ctx._barcode_debounce_timer.start()


