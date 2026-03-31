import numpy as np


def get_fixed_mic_page_channel_indices(total_channels, current_page, page_size=4):
    normalized_total = max(int(total_channels or 0), 0)
    normalized_page_size = max(int(page_size or 1), 1)
    if normalized_total == 0:
        return []

    max_page = max((normalized_total - 1) // normalized_page_size, 0)
    normalized_page = min(max(int(current_page or 0), 0), max_page)
    start = normalized_page * normalized_page_size
    end = min(start + normalized_page_size, normalized_total)
    return list(range(start, end))


def get_fixed_mic_downsample_step(visible_channel_count):
    normalized_count = max(int(visible_channel_count or 0), 0)
    if normalized_count <= 0:
        return 1
    if normalized_count <= 2:
        return 4
    return 8


def build_fixed_mic_plot_data(plot_audio, sample_rate, visible_channel_count):
    audio_data = np.asarray(plot_audio, dtype=np.float32)
    if audio_data.size == 0:
        return np.array([], dtype=np.float32), np.empty((0, 0), dtype=np.float32)

    if audio_data.ndim == 1:
        audio_data = audio_data.reshape(-1, 1)

    sample_count = audio_data.shape[0]
    if sample_count == 0:
        return np.array([], dtype=np.float32), np.empty((0, audio_data.shape[1]), dtype=np.float32)

    step = get_fixed_mic_downsample_step(visible_channel_count)
    sample_positions = np.arange(0, sample_count, step, dtype=np.int64)
    if sample_positions.size == 0 or sample_positions[-1] != sample_count - 1:
        sample_positions = np.append(sample_positions, sample_count - 1)

    display_audio = audio_data[sample_positions]
    time_axis = sample_positions.astype(np.float32) / float(sample_rate)
    return time_axis, display_audio


def get_fixed_mic_channel_titles(total_channels, channel_config):
    titles = []
    normalized_total = max(int(total_channels or 0), 0)
    channel_list = channel_config if isinstance(channel_config, list) else []

    for index in range(normalized_total):
        title = ""
        if index < len(channel_list) and isinstance(channel_list[index], dict):
            title = str(channel_list[index].get("label", "") or "").strip()
        titles.append(title or "Mic%d" % (index + 1))
    return titles
