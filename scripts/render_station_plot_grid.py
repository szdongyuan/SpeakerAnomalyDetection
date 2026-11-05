"""Render/check the real caller-fed grid without MainWindow, hardware or database.

Run from the repository root in separate processes for each font DPI::

    python -B scripts/render_station_plot_grid.py --output-dir output/station-plot-grid --font-dpi 96
    python -B scripts/render_station_plot_grid.py --output-dir output/station-plot-grid-120 --font-dpi 120
"""

import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--font-dpi', type=int, default=96)
    args = parser.parse_args()
    if args.font_dpi <= 0:
        parser.error('--font-dpi must be positive')
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if not output.is_relative_to(root / 'output'):
        parser.error('--output-dir must be inside this repository\'s output directory')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    os.environ['QT_FONT_DPI'] = str(args.font_dpi)
    sys.path.insert(0, str(root))

    import numpy as np
    from PyQt5.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt
    from PyQt5.QtGui import QFont, QFontDatabase, QImage, QPainter, QPalette, QPicture
    from PyQt5.QtSvg import QSvgRenderer
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication, QLabel, QScrollArea, QStyle, QStyleOptionComboBox, QWidget

    from consts.ui_style_const import COMBO_DOWN_ARROW_ICON, toolbar_combobox_style
    from ui.sequence.station_plot_cell import StationPlotCell
    from ui.sequence.station_plot_grid import StationPlotGrid

    app = QApplication.instance() or QApplication([])
    for name in ('msyh.ttc', 'msyhbd.ttc', 'seguisym.ttf'):
        path = Path('C:/Windows/Fonts') / name
        if QFontDatabase.addApplicationFont(str(path)) < 0:
            raise RuntimeError(f'Cannot register verification font: {path}')
    QFont.insertSubstitution('Microsoft YaHei UI', 'Segoe UI Symbol')
    app.setFont(QFont('Microsoft YaHei UI', 9))
    output.mkdir(parents=True, exist_ok=True)
    evidence = {'font_dpi': args.font_dpi, 'image_scale': 2, 'renders': []}

    def settle():
        # Multiple event turns allow QScrollArea geometry and pyqtgraph axes to settle.
        for _ in range(3):
            app.processEvents()
            QTest.qWait(20)

    def cells(grid):
        return sorted(grid.findChildren(StationPlotCell), key=lambda c: (c.row, c.column))

    def headers(grid):
        return [grid.findChild(QWidget, f'stationColumn{c}')
                for c in range(max(cell.column for cell in cells(grid)) + 1)]

    def state(grid):
        # Read-only inspection: preserve actual data, plot objects and opaque payload identities.
        return (
            [(id(cell), cell.kind, cell._configuration.copy(),
              [(s['x'].tobytes(), s['y'].tobytes(), s['name'], s['color']) for s in cell._series],
              None if cell._heatmap is None else cell._heatmap['values'].tobytes())
             for cell in cells(grid)],
            [(header.name.toolTip(), header.combo.currentIndex(),
              [(text, id(data)) for text, data in header.combo.options])
             for header in headers(grid)],
        )

    def populate(grid, rows, columns, *, long_text=False):
        grid.set_grid_size(rows, columns)
        for column in range(columns):
            suffix = ' — 长工位名称与完整下拉选项 / Long station and selection' if long_text else ''
            grid.set_station_name(column, f'工位 {column + 1}{suffix}')
            grid.set_column_options(column, [(f'检测 A{suffix}', {'slot': column}), ('检测 B', column)], 0)
            for row in range(rows):
                title = '时频概览' if (row + column) % 2 else 'SPL 趋势'
                right_text = f'MIC {row + 1}'
                if long_text:
                    title += ' — 很长的小图标题 / Full plot title retained in tooltip'
                    right_text += ' — 很长的麦克风说明 / Full microphone description retained in tooltip'
                subtexts = ('kHz' if (row + column) % 2 else 'dB(A)', '正转', '反转')
                if long_text:
                    subtexts = tuple(text + ' — 很长的中文 / Long <b>A & B</b>\nnext\tline' * 3 for text in subtexts)
                if (row + column) % 2:
                    grid.configure_plot(row, column, kind='heatmap', title=title, right_text=right_text,
                                        x_label='时间', x_unit='s', y_label='频率', y_unit='kHz')
                    # Distinct horizontal and vertical steps expose row-major/Y-low orientation.
                    matrix = np.arange(15, dtype=float).reshape(3, 5) + row + column
                    grid.set_heatmap_data(row, column, matrix,
                                          extent=(0, 5, 1, 4), levels=(0, 20))
                else:
                    grid.configure_plot(row, column, kind='curve', title=title, right_text=right_text,
                                        x_label='时间', x_unit='s', y_label='声压级', y_unit='dB(A)')
                    x = np.linspace(0, 2, 81)
                    other_x = np.linspace(0, 2, 57)
                    grid.set_curve_data(row, column, [
                        {'x': x, 'y': 65 + 3 * np.sin(x * np.pi * (column + 1)),
                         'name': '实测', 'color': '#2478C8'},
                        {'x': other_x, 'y': 65 + 1.8 * np.cos(other_x * np.pi * (row + 1)),
                         'name': '参考', 'color': '#E68A2E'},
                    ])

                grid.set_plot_subheader(row, column, *subtexts)

    def plot_span(cell):
        view = cell.plot_widget
        bounds = view.getViewBox().mapRectToScene(view.getViewBox().rect())
        offset = view.viewport().mapTo(cell, QPoint()).x() - cell.subheader.x()
        return [view.mapFromScene(QPointF(x, bounds.top())).x() + offset
                for x in (bounds.left(), bounds.right())]

    def subheader_evidence(cell):
        bar = cell.subheader
        labels = (bar.left, bar.first_half, bar.second_half)
        return {'texts': [label.toolTip() for label in labels],
                'display': [label.text() for label in labels],
                'visible': bar.isVisible(), 'rect': rect(bar, cell),
                'label_rects': [rect(label, bar) for label in labels],
                'viewbox_span': plot_span(cell)}

    def rect(widget, ancestor):
        point = widget.mapTo(ancestor, QPoint(0, 0))
        return [point.x(), point.y(), widget.width(), widget.height()]

    def save(grid, scenario, *, preserved=True):
        settle()
        body = grid.findChild(QScrollArea, 'stationPlotScroll')
        header_scroll = grid.findChild(QScrollArea, 'stationHeaderScroll')
        all_cells = cells(grid)
        column_controls = headers(grid)
        marks_visible = column_controls[0].ok.isVisible()
        checks = {'column_alignment': True, 'controls_fit': True,
                  'marks_no_placeholder': True, 'state_preserved': preserved,
                  'plot_contents_fit': True, 'corner_header_geometry': True,
                  'corner_text_elision': True, 'corner_text_style': True,
                  'subheader_geometry': True, 'subheader_text': True,
                  'subheader_viewbox_alignment': True,
                  # QScrollArea/viewport regions can be empty when opaque children
                  # cover them; inspect the actual visible column widgets instead.
                  'fixed_header_visible': (any(not h.visibleRegion().isEmpty() for h in column_controls)
                                           and header_scroll.mapTo(grid, QPoint()).y()
                                           < body.mapTo(grid, QPoint()).y())}
        columns = []
        for cell in all_cells:
            view = cell.plot_widget
            plot = view.getPlotItem()
            header = cell.header
            left, right = cell._configuration['title'], cell._configuration['right_text']
            checks['corner_header_geometry'] &= (
                header.isVisible() == bool(left or right)
                and header.left.isVisible() == bool(left)
                and header.right.isVisible() == bool(right)
                and not plot.titleLabel.isVisible())
            if left or right:
                checks['corner_header_geometry'] &= (
                    cell.rect().contains(header.geometry())
                    and header.geometry().bottom() < view.y())
                visible_labels = [label for label in (header.left, header.right) if label.isVisible()]
                checks['corner_header_geometry'] &= all(
                    header.rect().contains(label.geometry()) for label in visible_labels)
                if left and right:
                    checks['corner_header_geometry'] &= (
                        not header.left.geometry().intersects(header.right.geometry())
                        and abs(header.left.width() - 2 * header.right.width()) <= 2)
                else:
                    checks['corner_header_geometry'] &= (
                        visible_labels[0].width() == header.width() - 16)
            else:
                checks['corner_header_geometry'] &= view.y() == (cell.subheader.height() if cell.subheader.isVisible() else 0)
            for label, text, color, alignment in (
                    (header.left, left, '#183957', Qt.AlignLeft),
                    (header.right, right, '#5d7691', Qt.AlignRight)):
                checks['corner_text_elision'] &= (
                    label.toolTip() == text and label.textFormat() == Qt.PlainText
                    and label.text() == label.fontMetrics().elidedText(
                        ' '.join(text.splitlines()).replace('\t', ' '),
                        Qt.ElideRight, label.contentsRect().width()))
                checks['corner_text_style'] &= (
                    label.palette().color(QPalette.WindowText).name() == color
                    and label.font().pixelSize() == 13 and bool(label.alignment() & alignment))
            sub = cell.subheader
            labels = (sub.left, sub.first_half, sub.second_half)
            texts = [cell._configuration[key] for key in (
                'subheader_left_text', 'subheader_first_half_text', 'subheader_second_half_text')]
            checks['subheader_geometry'] &= sub.isVisible() == any(texts)
            checks['subheader_geometry'] &= all(label.isVisible() == bool(text) for label,text in zip(labels,texts))
            if any(texts):
                checks['subheader_geometry'] &= (
                    cell.rect().contains(sub.geometry()) and sub.geometry().bottom() < view.y()
                    and (not header.isVisible() or header.geometry().bottom() < sub.y())
                    and all(sub.rect().contains(label.geometry()) for label in labels)
                    and all(not a.geometry().intersects(b.geometry())
                            for i,a in enumerate(labels) for b in labels[i+1:]))
                left,right = plot_span(cell)
                checks['subheader_viewbox_alignment'] &= sub.left.geometry().right() < left
                checks['subheader_viewbox_alignment'] &= all(
                    abs(label.geometry().center().x() - (left+(right-left)*fraction)) <= 2
                    for label,fraction in zip(labels[1:],(.25,.75)))
            for label,text in zip(labels,texts):
                checks['subheader_text'] &= (
                    label.toolTip() == text and label.textFormat() == Qt.PlainText
                    and label.font().pixelSize() == 11
                    and label.palette().color(QPalette.WindowText).name() == '#5d7691'
                    and label.text() == label.fontMetrics().elidedText(
                        ' '.join(text.splitlines()).replace('\t',' '), Qt.ElideRight, label.width()))
            checks['plot_contents_fit'] &= plot.geometry().right() <= view.viewport().width()
            for name in ('bottom', 'left'):
                item = plot.getAxis(name)
                bounds = view.mapFromScene(item.mapRectToScene(item.rect())).boundingRect()
                checks['plot_contents_fit'] &= (
                    bounds.left() >= 0 and bounds.right() <= view.viewport().width()
                    and bounds.top() >= 0 and bounds.bottom() <= view.viewport().height())
            axis = plot.getAxis('bottom')
            picture = QPicture()
            painter = QPainter(picture)
            try:
                if axis.style['tickFont'] is not None:
                    painter.setFont(axis.style['tickFont'])
                specs = axis.generateDrawSpecs(painter)
            finally:
                painter.end()
            if specs is not None:
                viewport = view.mapToScene(view.viewport().rect()).boundingRect()
                checks['plot_contents_fit'] &= all(
                    viewport.contains(axis.mapRectToScene(text_rect))
                    for text_rect, flags, text in specs[2])
            if cell._colorbar is not None:
                bar = cell._colorbar
                bounds = view.mapFromScene(bar.mapRectToScene(bar.rect())).boundingRect()
                checks['plot_contents_fit'] &= (
                    bounds.left() >= 0 and bounds.right() <= view.viewport().width()
                    and bounds.top() >= 0 and bounds.bottom() <= view.viewport().height())
        for column, header in enumerate(column_controls):
            header_rect = rect(header, grid)
            for cell in (c for c in all_cells if c.column == column):
                cell_rect = rect(cell, grid)
                checks['column_alignment'] &= (header_rect[0] == cell_rect[0]
                                               and header_rect[2] == cell_rect[2])
            visible = [w for w in (header.name, header.combo, header.play, header.ok, header.ng)
                       if w.isVisible()]
            checks['controls_fit'] &= all(header.rect().contains(w.geometry()) for w in visible)
            checks['controls_fit'] &= all(
                not left.geometry().intersects(right.geometry())
                for i, left in enumerate(visible) for right in visible[i + 1:])
            # When marks are hidden, play occupies the old NG position: no retained space.
            checks['marks_no_placeholder'] &= (
                marks_visible or (not header.ok.isVisible() and not header.ng.isVisible()
                                  and header.width() - header.play.geometry().right() <= 8))
            columns.append({'column': column, 'header_rect': header_rect,
                            'current_index': header.combo.currentIndex(),
                            'option_count': header.combo.count(),
                            'control_rects': {w.objectName(): rect(w, header) for w in visible}})
        scroll = {}
        for name, bar in [('horizontal', body.horizontalScrollBar()),
                          ('vertical', body.verticalScrollBar())]:
            scroll[name] = {'value': bar.value(), 'maximum': bar.maximum()}
        checks['horizontal_sync'] = (body.horizontalScrollBar().value()
                                     == header_scroll.horizontalScrollBar().value())
        title = grid.findChild(QLabel, 'stationPlotTitle')
        title_rect = rect(title, grid)
        header_rect = rect(header_scroll, grid)
        body_rect = rect(body, grid)
        rows = max(cell.row for cell in all_cells) + 1
        count = len(column_controls)
        by_position = {(cell.row, cell.column): cell for cell in all_cells}
        checks['continuous_grid'] = (
            title_rect[0:2] == [1, 1]
            and title_rect[1] + title_rect[3] == header_rect[1]
            and header_rect[1] + header_rect[3] == body_rect[1]
            and body.widget().width() == max(count * 280, body.viewport().width())
            and body.widget().height() == max(rows * 180, body.viewport().height())
            and all(cell.geometry().right() + 1 == by_position[cell.row, cell.column + 1].x()
                    for cell in all_cells if cell.column < count - 1)
            and all(cell.geometry().bottom() + 1 == by_position[cell.row + 1, cell.column].y()
                    for cell in all_cells if cell.row < rows - 1))
        checks['painted_title'] = (
            title.height() >= 36
            and title.grab().toImage().pixelColor(title.width() - 3, 3).name() == '#3e75ac'
            and title.palette().color(QPalette.WindowText).name() == '#ffffff')
        checks['painted_station_background'] = all(
            h.height() == 40 and h.grab().toImage().pixelColor(2, 2).name() == '#f4f7fc'
            and h.name.palette().color(QPalette.WindowText).name() == '#183957'
            for h in column_controls)
        checks['single_grid_separators'] = True
        for cell in all_cells:
            image = cell.grab().toImage()
            right = '#cddaea' if cell.column < count - 1 else '#ffffff'
            bottom = '#cddaea' if cell.row < rows - 1 else '#ffffff'
            checks['single_grid_separators'] &= (
                image.pixelColor(cell.width() - 1, 2).name() == right
                and image.pixelColor(cell.width() - 2, 2).name() == '#ffffff'
                and image.pixelColor(0, 2).name() == '#ffffff'
                # Sample in the blank corner: bottom-axis text can legitimately
                # reach the final plot pixel at larger font DPI.
                and image.pixelColor(2, cell.height() - 1).name() == bottom
                and image.pixelColor(2, cell.height() - 2).name() == '#ffffff')
        for column, header in enumerate(column_controls):
            image = header.grab().toImage()
            right = '#cddaea' if column < count - 1 else '#f4f7fc'
            checks['single_grid_separators'] &= (
                image.pixelColor(header.width() - 1, 2).name() == right
                and image.pixelColor(header.width() - 2, 2).name() == '#f4f7fc'
                and image.pixelColor(0, 2).name() == '#f4f7fc'
                and image.pixelColor(2, 39).name() == '#cddaea'
                and image.pixelColor(2, 38).name() == '#f4f7fc')
        frame = grid.grab().toImage()
        checks['closed_outer_frame'] = (
            all(frame.pixelColor(x, y).name() == '#cddaea'
                for x in range(grid.width()) for y in (0, grid.height() - 1))
            and all(frame.pixelColor(x, y).name() == '#cddaea'
                    for x in (0, grid.width() - 1) for y in range(grid.height())))
        checks['project_combo_arrow'] = QSvgRenderer(COMBO_DOWN_ARROW_ICON).isValid()
        for header in column_controls:
            box = header.combo
            option = QStyleOptionComboBox()
            box.initStyleOption(option)
            arrow = box.style().subControlRect(
                QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxArrow, box).center()
            image = box.grab().toImage()
            checks['project_combo_arrow'] &= (
                toolbar_combobox_style in box.styleSheet()
                and any(image.pixelColor(x, y).lightness() < 250
                        for x in range(arrow.x() - 4, arrow.x() + 6)
                        for y in range(arrow.y() - 4, arrow.y() + 6)))
        if scenario == 'narrow-end':
            last = all_cells[-1]
            checks['last_column_reachable'] = (
                scroll['horizontal']['value'] == scroll['horizontal']['maximum'] > 0
                and rect(last, body.viewport())[0] + last.width() == body.viewport().width())
        if scenario == 'short-end':
            last = all_cells[-1]
            checks['last_row_reachable'] = (
                scroll['vertical']['value'] == scroll['vertical']['maximum'] > 0
                and rect(last, body.viewport())[1] + last.height() == body.viewport().height())
        assert all(checks.values()), (scenario, checks, columns, scroll)
        image = QImage(grid.width() * 2, grid.height() * 2, QImage.Format_ARGB32)
        image.setDevicePixelRatio(2)
        image.fill(Qt.transparent)
        painter = QPainter(image)
        grid.render(painter)
        painter.end()
        target = output / f'{scenario}.png'
        if not image.save(str(target)):
            raise RuntimeError(f'Cannot save {target}')
        evidence['renders'].append({
            'scenario': scenario, 'file': target.name, 'size': [grid.width(), grid.height()],
            'grid_size': [max(c.row for c in all_cells) + 1, len(column_controls)],
            'marks_visible': marks_visible, 'scroll': scroll, 'checks': checks, 'columns': columns,
            'cells': [{'row': c.row, 'column': c.column, 'kind': c.kind, 'rect': rect(c, grid),
                       'curve_count': len(c.plot_widget.getPlotItem().listDataItems()),
                       'header': {'left_text': c.header.left.toolTip(),
                                  'right_text': c.header.right.toolTip(),
                                  'left_display': c.header.left.text(),
                                  'right_display': c.header.right.text(),
                                  'visible': c.header.isVisible(),
                                  'rect': rect(c.header, c),
                                  'left_rect': rect(c.header.left, c.header),
                                  'right_rect': rect(c.header.right, c.header)},
                       'subheader': subheader_evidence(c),
                       'y_axis_text': c.plot_widget.getAxis('left').label.toPlainText(),
                       'heatmap_extent': None if c._heatmap is None else list(c._heatmap['extent']),
                       'heatmap_shape': None if c._heatmap is None else list(c._heatmap['values'].shape)}
                      for c in all_cells],
        })
        print(f'{target.name}: {grid.width()}x{grid.height()}, checks passed')

    def create(width, height):
        grid = StationPlotGrid()
        grid.resize(width, height)
        grid.show()
        settle()
        assert (grid.width(), grid.height()) == (width, height)
        return grid

    def dispose(grid):
        grid.close()
        grid.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    grid = create(1100, 650)
    assert len(cells(grid)) == 6
    assert all(not c._series and c._heatmap is None for c in cells(grid))
    assert all(not header.combo.isEnabled() for header in headers(grid))
    save(grid, 'default-empty')
    populate(grid, 2, 3)
    save(grid, 'populated-2x3')
    dispose(grid)

    grid = create(1240, 800)
    populate(grid, 3, 4)
    save(grid, 'mixed-3x4')
    before = state(grid)
    grid.set_mark_buttons_visible(False)
    save(grid, 'marks-hidden', preserved=state(grid) == before)
    grid.set_mark_buttons_visible(True)
    save(grid, 'marks-restored', preserved=state(grid) == before)
    dispose(grid)

    for prefix, width, height, axis in [('narrow', 640, 760, 'horizontal'),
                                        ('short', 1240, 360, 'vertical')]:
        grid = create(width, height)
        populate(grid, 3, 4)
        save(grid, f'{prefix}-start')
        before = state(grid)
        body = grid.findChild(QScrollArea, 'stationPlotScroll')
        bar = body.horizontalScrollBar() if axis == 'horizontal' else body.verticalScrollBar()
        assert bar.maximum() > 0
        bar.setValue(bar.maximum())
        save(grid, f'{prefix}-end', preserved=state(grid) == before)
        dispose(grid)

    grid = create(900, 650)
    populate(grid, 2, 3, long_text=True)
    save(grid, 'long-text')
    assert all(len(header.name.toolTip()) > len(header.name.text()) for header in headers(grid))
    assert all(c.header.left.text() != c.header.left.toolTip()
               and c.header.right.text() != c.header.right.toolTip() for c in cells(grid))
    dispose(grid)

    grid = create(900, 650)
    populate(grid, 2, 3)
    for cell in cells(grid):
        title = 'SPL 趋势' if cell.kind == 'curve' else '时频分布'
        left, right = [(title, ''), ('', 'MIC 2'), ('', '')][cell.column]
        grid.set_plot_header(cell.row, cell.column, left, right)
    save(grid, 'header-variants')
    dispose(grid)
    grid = create(900, 650)
    populate(grid, 2, 3)
    for cell in cells(grid):
        # Exercise each independent row and empty fixed slots.
        if cell.row == 0:
            grid.set_plot_header(cell.row, cell.column)
        grid.set_plot_subheader(cell.row, cell.column,
                               *[(('kHz' if cell.kind == 'heatmap' else 'dB(A)'), '', '反转'),
                                 ('', '正转', ''), ('', '', '')][cell.column])
    save(grid, 'subheader-variants')
    dispose(grid)

    grid = create(562, 258)
    populate(grid, 1, 2, long_text=True)
    # Keep this minimum-size case focused on both text rows and axis/colorbar
    # bounds; the existing two-row legend needs a taller ViewBox at 120 DPI.
    grid.set_curve_data(0, 0, [{'x': [0, 1, 2], 'y': [65, 68, 66], 'name': '实测'}])
    save(grid, 'minimum-subheaders')
    assert all((cell.width(), cell.height()) == (280, 180) for cell in cells(grid))
    dispose(grid)

    grid = create(1100, 650)
    populate(grid, 2, 3)
    for cell in cells(grid):
        cell.plot_widget.getAxis('left').setWidth(45 + cell.column * 20)
        if cell.kind == 'heatmap' and cell.row == 1:
            cell.set_heatmap_data(None)
    save(grid, 'axis-widths')
    dispose(grid)
    (output / 'render-evidence.json').write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
