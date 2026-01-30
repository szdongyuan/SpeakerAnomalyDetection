# Excel 同步逻辑分析

## 修改的三个关键点

### 1. `main_window.py::closeEvent` (关闭主窗口)
**原逻辑:**
```python
try:
    if hasattr(self, "sequence_window") and self.sequence_window is not None:
        self.sequence_window.flush_excel_spool_build(on_close=False)
except Exception:
    pass  # 失败直接忽略
event.accept()
```

**新逻辑:**
```python
if hasattr(self, "sequence_window") and self.sequence_window is not None:
    while True:
        failures = self.sequence_window.flush_excel_spool_build(on_close=False)
        if not failures:
            break  # 成功则退出循环

        # 弹窗让用户选择
        [显示对话框: 重试 / 忽略]

        if 用户点击重试:
            continue  # 重新尝试
        else:
            break  # 用户点忽略，退出循环
event.accept()
```

### 2. `sequence_widget.py::closeEvent` (关闭 SequenceWindow)
**原逻辑:**
```python
self.flush_excel_spool_build(on_close=True)  # 失败不管，继续
if hasattr(self, "hw_manager"):
    self.hw_manager.stop()
super().closeEvent(event)
```

**新逻辑:**
```python
while True:
    failures = self.flush_excel_spool_build(on_close=True)
    if not failures:
        break

    [显示对话框: 重试 / 忽略]

    if 用户点击重试:
        continue
    else:
        break  # 用户点忽略

if hasattr(self, "hw_manager"):
    self.hw_manager.stop()
super().closeEvent(event)
```

### 3. `sequence_widget.py::_maybe_export_excel_results` (录音后保存)
**原逻辑:**
```python
for cfg_name, excel_cfg in excel_cfg_list:
    ret = export_to_xxx(...)
    if ret.ok:
        log success
    else:
        all_ok = False
        log error
        if show_message:  # 只在手动点击OK/NG时为True
            QMessageBox.warning(self, "Excel保存失败", ret.message)

if all_ok:
    self._excel_exported_record_id = record_id
    if spool_cfgs:
        self._schedule_excel_spool_build(spool_cfgs)
# 失败就失败了，不重试
```

**新逻辑:**
```python
while True:
    failed_exports = []
    for cfg_name, excel_cfg in excel_cfg_list:
        ret = export_to_xxx(...)
        if not ret.ok:
            failed_exports.append((cfg_name, ret.message))

    if not failed_exports:
        # 全部成功
        self._excel_exported_record_id = record_id
        if spool_cfgs:
            self._schedule_excel_spool_build(spool_cfgs)
        break

    # 有失败，弹窗
    [显示对话框: 重试 / 忽略]

    if 用户点击重试:
        continue  # 重新尝试所有导出
    else:
        break  # 忽略，数据丢失
```

## 潜在问题检查

### ✅ 1. 点"忽略"是否改变原逻辑？
**结论：NO，不会改变**

- **关闭时点忽略**：和原来的 `except: pass` 一样，失败了就继续关闭
- **录音后点忽略**：和原来的失败逻辑一样，数据丢失但程序继续

### ✅ 2. 是否会死锁？
**结论：NO，不会死锁**

**原因:**
1. 所有 `while True` 都有 `break` 出口
2. 对话框 `msg_box.exec_()` 是模态阻塞，但用户操作后必然返回
3. 用户点"重试"或"忽略"都会 `break`，不会无限循环
4. 没有等待外部锁或资源的逻辑

**可能的"卡住"场景（非死锁）:**
- 用户一直点"重试"但不关闭 Excel 文件 → 这是用户行为，不是代码bug
- 解决方法：用户可以随时点"忽略"跳出

### ✅ 3. 关闭时不点"忽略"是否一定会同步？
**结论：YES，会一直重试直到成功或用户点忽略**

**流程:**
```
关闭程序
  ↓
flush_excel_spool_build() 执行
  ↓
失败？
  ├─ NO → 同步成功 → 关闭
  └─ YES → 弹窗
           ├─ 点"重试" → 重新执行 flush_excel_spool_build()
           └─ 点"忽略" → 放弃同步，关闭
```

### ⚠️ 4. 新增的潜在问题

#### 4.1 `flush_excel_spool_build` 的发现逻辑改变
**原逻辑:** 依赖 `_excel_spool_build_pending_cfgs` 列表
**新逻辑:** 从 `analysis_config` 重新扫描所有 Excel 配置

**风险:**
- 如果 `analysis_config` 为空或未初始化，会返回空列表（`failures = []`）
- 这会导致关闭时认为"没有失败"，直接退出
- **但这和原逻辑一致**：如果 `pending_cfgs` 为空，原来也是直接 return

#### 4.2 CSV 已写入但 Excel 未同步的场景
**场景:**
1. 用户录音后，CSV 写入成功（`export_analysis_to_csv_spool` 成功）
2. 用户点"忽略"，Excel 未构建（`_schedule_excel_spool_build` 被调用但标记为 pending）
3. 关闭时，`flush_excel_spool_build` 会尝试同步这些 pending 的配置

**结论:** ✅ 这是好的，关闭时会兜底

#### 4.3 "快速跳过"逻辑可能误判
```python
# build_excel_from_csv_spool 中的逻辑
if os.path.exists(xlsx_path) and os.path.getmtime(xlsx_path) >= latest_csv_mtime:
    return ExportResult(ok=True, message="Excel已是最新")
```

**风险场景:**
1. Excel 文件存在且时间戳比 CSV 新
2. 但实际上这次录音的数据还没写入 Excel
3. 会被跳过，认为"已是最新"

**分析:**
- 这个问题在原代码中就存在
- 但由于 `export_analysis_to_csv_spool` 会更新 CSV 的 mtime，所以关闭时 CSV 应该比 Excel 新
- ✅ 逻辑应该正常工作

### ✅ 5. 是否会多次弹窗？
**结论：NO**

- 关闭主窗口时：只调用一次 `main_window.closeEvent`
- 关闭 SequenceWindow 时：只调用一次 `sequence_widget.closeEvent`
- 录音后保存时：只在当前录音失败时弹一次窗

## 改进建议（可选）

### 建议 1: 添加日志记录用户操作
```python
if msg_box.clickedButton() == retry_btn:
    self.default_logger.info("用户选择重试同步")
    continue
else:
    self.default_logger.warning("用户忽略同步失败，数据可能丢失")
    break
```

### 建议 2: 限制重试次数（避免用户无限点重试）
```python
retry_count = 0
max_retries = 10  # 最多重试10次
while retry_count < max_retries:
    failures = ...
    if not failures:
        break

    retry_count += 1
    if retry_count >= max_retries:
        QMessageBox.critical(self, "错误", "重试次数过多，请检查文件权限或联系技术支持")
        break

    [显示对话框]
```

### 建议 3: 在对话框中显示文件路径
帮助用户知道要关闭哪个文件：
```python
msg_box.setText(f"无法同步到Excel文件：\n{xlsx_path}\n\n请关闭该文件后重试。")
```

## 总结

### ✅ 安全性
- **不会死锁**: 所有循环都有明确的退出条件
- **不会改变原逻辑**: 点"忽略"等同于原来的失败行为
- **一定会尝试同步**: 关闭时会强制尝试同步，除非用户明确忽略

### ✅ 用户体验
- **友好提示**: 明确告知用户失败原因
- **可控操作**: 用户可以选择重试或忽略
- **数据安全**: 减少了数据丢失的可能性

### ⚠️ 注意事项
1. 用户可能会被弹窗"骚扰"，如果经常忘记关闭 Excel
2. 如果用户一直点重试但不操作，程序会一直等待
3. `analysis_config` 必须正确初始化，否则关闭时不会弹窗（但也不会崩溃）

### 推荐的额外测试场景
1. ✅ CSV 写入失败（文件被占用）→ 录音后弹窗
2. ✅ Excel 文件被占用 → 关闭时弹窗
3. ✅ 用户点"忽略" → 程序正常关闭
4. ✅ 用户点"重试"并关闭Excel → 同步成功并关闭
5. ⚠️ `analysis_config` 为空 → 关闭时不弹窗（应该测试）
6. ⚠️ 网络驱动器或权限问题 → 确保错误消息清晰
