"""Isolated template layouts; never load or change the user's UI configuration."""


def template_layout():
    return {
        'rows': 1, 'columns': 4,
        'default_plot_style': {
            'curve_line_width': 1.5,
            'channel_curve_styles': [
                {'wav_channel': 1, 'curve_color': '#33C44D'},
                {'wav_channel': 2, 'curve_color': '#3B82F6', 'curve_line_style': 'dashed'},
            ],
            'grid': {'visible': True, 'color': '#E5EAF0', 'line_style': 'dotted', 'line_width': 1.0},
        },
        'default_condition_display': {'layout': 'single_canvas', 'condition_source': 'current_configuration', 'x_axis_mode': 'cumulative_time', 'condition_gap_px': 0, 'show_condition_names': True, 'show_condition_separator': True},
        'plot_templates': {'spl': {
            'title': '声压级', 'analysis_item_name': '声压级', 'analysis_type': 'SPL', 'chart_type': 'line',
            'channel_display': {'show_all_analyzed_channels': True},
            'plot_style': {'curve_line_width': 2.0},
            'x_axis': {'label': '时间', 'scale': 'linear', 'auto_range': False, 'range_min': 0, 'range_max': 3},
            'y_axis': {'label': '声压级', 'scale': 'linear', 'auto_range': True},
        }},
        'cells': [[{'plot_template': 'spl'} for _ in range(4)]],
    }
