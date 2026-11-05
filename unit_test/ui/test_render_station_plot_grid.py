"""Exercise the standalone visual acceptance CLI through its real process boundary."""

import json
from pathlib import Path
import subprocess
import sys

from PyQt5.QtGui import QImage


def test_render_cli_produces_checked_real_grid_scenarios(tmp_path):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / 'renders'
    result = subprocess.run(
        [sys.executable, '-B', 'scripts/render_station_plot_grid.py',
         '--output-dir', str(output), '--font-dpi', '96'],
        cwd=root, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads((output / 'render-evidence.json').read_text(encoding='utf-8'))
    assert evidence['font_dpi'] == 96
    assert evidence['image_scale'] == 2
    renders = {entry['scenario']: entry for entry in evidence['renders']}
    assert set(renders) == {
        'default-empty', 'populated-2x3', 'mixed-3x4',
        'narrow-start', 'narrow-end', 'short-start', 'short-end',
        'marks-hidden', 'marks-restored', 'long-text', 'header-variants',
        'subheader-variants', 'minimum-subheaders', 'axis-widths',
    }
    assert all(cell['curve_count'] == 0 and cell['heatmap_shape'] is None
               for cell in renders['default-empty']['cells'])
    assert renders['mixed-3x4']['grid_size'] == [3, 4]
    assert any(cell['heatmap_shape'] == [3, 5]
               for cell in renders['mixed-3x4']['cells'])
    for axis, end in [('horizontal', 'narrow-end'), ('vertical', 'short-end')]:
        scroll = renders[end]['scroll'][axis]
        assert scroll['value'] == scroll['maximum'] > 0
    assert renders['marks-hidden']['marks_visible'] is False
    assert renders['marks-restored']['marks_visible'] is True
    assert all(cell['header']['left_text'] == '' and cell['header']['right_text'] == ''
               and not cell['header']['visible'] for cell in renders['default-empty']['cells'])
    assert any(cell['header']['left_text'] == 'SPL 趋势' and cell['header']['right_text'] == 'MIC 1'
               for cell in renders['populated-2x3']['cells'])
    for cell in renders['long-text']['cells']:
        header = cell['header']
        assert header['left_text'] != header['left_display']
        assert header['right_text'] != header['right_display']
    assert {(cell['header']['left_text'], cell['header']['right_text'])
            for cell in renders['header-variants']['cells']} == {
                ('SPL 趋势', ''), ('时频分布', ''), ('', 'MIC 2'), ('', '')}
    typical = renders['populated-2x3']['cells']
    assert any(c['kind'] == 'curve' and c['subheader']['texts'] == ['dB(A)', '正转', '反转'] for c in typical)
    assert any(c['kind'] == 'heatmap' and c['subheader']['texts'] == ['kHz', '正转', '反转'] for c in typical)
    for cell in typical:
        if cell['kind'] == 'heatmap':
            assert cell['heatmap_extent'] == [0, 5, 1, 4]
            assert cell['y_axis_text'] == '频率 (kHz)'
    for cell in renders['default-empty']['cells']:
        assert cell['subheader']['texts'] == ['', '', ''] and not cell['subheader']['visible']
    variants = renders['subheader-variants']['cells']
    assert any(c['subheader']['visible'] and not c['header']['visible'] for c in variants)
    assert any(not c['subheader']['visible'] and c['header']['visible'] for c in variants)
    assert any(c['subheader']['texts'] == ['', '正转', ''] for c in variants)
    minimum = renders['minimum-subheaders']['cells']
    assert all(c['rect'][2:] == [280, 180] for c in minimum)
    assert next(c for c in minimum if c['kind'] == 'curve')['curve_count'] == 1
    assert all(c['header']['visible'] and c['subheader']['visible'] for c in minimum)
    for cell in renders['long-text']['cells']:
        assert all(a != b for a,b in zip(cell['subheader']['texts'],cell['subheader']['display']))
    assert any(c['kind'] == 'heatmap' and c['heatmap_shape'] is None for c in renders['axis-widths']['cells'])
    for entry in renders.values():
        assert {'continuous_grid', 'painted_title', 'painted_station_background',
                'single_grid_separators', 'closed_outer_frame', 'project_combo_arrow',
                'station_control_dimensions',
                'corner_header_geometry', 'corner_text_elision', 'corner_text_style',
                'subheader_geometry', 'subheader_text', 'subheader_viewbox_alignment'} <= entry['checks'].keys()
        assert all(entry['checks'].values()), entry
        for cell in entry['cells']:
            sub = cell['subheader']
            if sub['visible']:
                left,right = sub['viewbox_span']
                for bounds,fraction in zip(sub['label_rects'][1:],(.25,.75)):
                    assert abs(bounds[0]+(bounds[2]-1)/2-(left+(right-left)*fraction)) <= 2
        image = QImage(str(output / entry['file']))
        assert not image.isNull()
        assert [image.width(), image.height()] == [v * 2 for v in entry['size']]
