
FEATURE_CONFIG = {
    "waveform": {
        "display_name": "原始波形 (Waveform)",
        "params": {}
    },
    "fft": {
        "display_name": "快速傅里叶变换 (FFT)",
        "params": {}
    },
    "mfcc": {
        "display_name": "梅尔倒谱系数 (MFCC)",
        "params": {
            "n_mfcc": {
                "label": "MFCC系数数量 (n_mfcc):",
                "type": "int",
                "default": 20,
                "validation": {"type": "int", "min": 10, "max": 80}
            },
            "n_fft": {
                "label": "FFT窗口大小 (n_fft):",
                "type": "int",
                "default": 2048,
                "validation": {"type": "int", "min": 128, "max": 8192}
            },
            "hop_length": {
                "label": "帧移 (hop_length):",
                "type": "int",
                "default": 512,
                "validation": {"type": "int", "min": 64, "max": 4096}
            },
            "win_length": {
                "label": "窗长 (win_length):",
                "type": "int",
                "default": 2048,
                "validation": {"type": "int", "min": 128, "max": 8192}
            },
            "window": {
                "label": "窗函数 (window):",
                "type": "dropdown",
                "default": "hann",
                "options": [
                    {"display_name": "汉宁窗 (Hann)", "value": "hann"},
                    {"display_name": "海明窗 (Hamming)", "value": "hamming"},
                    {"display_name": "矩形窗 (Rectangular)", "value": "boxcar"}
                ]
            },
            "n_mels": {
                "label": "梅尔带数量 (n_mels):",
                "type": "int",
                "default": 128,
                "validation": {"type": "int", "min": 32, "max": 512}
            },
            "fmin": {
                "label": "最低频率 (fmin Hz):",
                "type": "int",
                "default": 0,
                "validation": {"type": "int", "min": 0, "max": 22050}
            },
            "fmax": {
                "label": "最高频率 (fmax Hz):",
                "type": "int",
                "default": 8000,
                "validation": {"type": "int", "min": 100, "max": 22050}
            }
        }
    },
    "spec": {
        "display_name": "频谱图 (Spectrogram)",
        "params": {
            "n_fft": {
                "label": "FFT窗口大小 (n_fft):",
                "type": "int",
                "default": 2048,
                "validation": {"type": "int", "min": 128, "max": 8192}
            },
            "hop_length": {
                "label": "帧移 (hop_length):",
                "type": "int",
                "default": 512,
                "validation": {"type": "int", "min": 64, "max": 4096}
            },
            "win_length": {
                "label": "窗长 (win_length):",
                "type": "int",
                "default": 2048,
                "validation": {"type": "int", "min": 128, "max": 8192}
            },
            "window": {
                "label": "窗函数 (window):",
                "type": "dropdown",
                "default": "hann",
                "options": [
                    {"display_name": "汉宁窗 (Hann)", "value": "hann"},
                    {"display_name": "海明窗 (Hamming)", "value": "hamming"},
                    {"display_name": "矩形窗 (Rectangular)", "value": "boxcar"}
                ]
            },
            "power": {
                "label": "能量度:",
                "type": "float",
                "default": 2.0,
                "validation": {"type": "float", "min": 1.0, "max": 2.0, "decimals": 1}
            },
            "power_to_db": {
                "label": "转换为分贝(dB):",
                "type": "bool",
                "default": True
            }
        }
    },
    "melspec": {
        "display_name": "梅尔频谱图 (Mel Spectrogram)",
        "params": {
            "n_fft": {
                "label": "FFT窗口大小 (n_fft):",
                "type": "int",
                "default": 2048,
                "validation": {"type": "int", "min": 128, "max": 8192}
            },
            "hop_length": {
                "label": "帧移 (hop_length):",
                "type": "int",
                "default": 512,
                "validation": {"type": "int", "min": 64, "max": 4096}
            },
            "win_length": {
                "label": "窗长 (win_length):",
                "type": "int",
                "default": 2048,
                "validation": {"type": "int", "min": 128, "max": 8192}
            },
            "window": {
                "label": "窗函数 (window):",
                "type": "dropdown",
                "default": "hann",
                "options": [
                    {"display_name": "汉宁窗 (Hann)", "value": "hann"},
                    {"display_name": "海明窗 (Hamming)", "value": "hamming"},
                    {"display_name": "矩形窗 (Rectangular)", "value": "boxcar"}
                ]
            },
            "n_mels": {
                "label": "梅尔带数量 (n_mels):",
                "type": "int",
                "default": 128,
                "validation": {"type": "int", "min": 32, "max": 512}
            },
            "fmin": {
                "label": "最低频率 (fmin Hz):",
                "type": "int",
                "default": 0,
                "validation": {"type": "int", "min": 0, "max": 22050}
            },
            "fmax": {
                "label": "最高频率 (fmax Hz):",
                "type": "int",
                "default": 8000,
                "validation": {"type": "int", "min": 100, "max": 22050}
            },
            "power": {
                "label": "能量度:",
                "type": "float",
                "default": 1.0,
                "validation": {"type": "float", "min": 1.0, "max": 2.0, "decimals": 1}
            },
            "power_to_db": {
                "label": "转换为分贝(dB):",
                "type": "bool",
                "default": True
            }
        }
    }
}


ALGORITHM_CONFIG = {
    "distance": {
        "display_name": "距离度量 (DM)",
        "params": {
            "metric": {
                "label": "度量方法:",
                "type": "dropdown",
                "default": "euclidean",
                "options": [
                    {"display_name": "欧氏距离 (Euclidean)", "value": "euclidean"},
                    {"display_name": "余弦相似度 (Cosine)", "value": "cosine"},
                    {"display_name": "曼哈顿距离 (Manhattan)", "value": "manhattan"}
                ]
            }
        }
    },
    "dtw": {
        "display_name": "动态时间规整 (DTW)",
        "params": {
            "metric": {
                "label": "距离度量:",
                "type": "dropdown",
                "default": "euclidean",
                "options": [
                    {"display_name": "欧氏距离 (Euclidean)", "value": "euclidean"},
                    {"display_name": "余弦相似度 (Cosine)", "value": "cosine"}
                ]
            },
            "normalization": {
                "label": "路径归一化:",
                "type": "dropdown",
                "default": "path_length",
                "options": [
                    {"display_name": "按路径长度 (By Path Length)", "value": "path_length"},
                    {"display_name": "按序列长度和 (By N+M)", "value": "sum_length"},
                    {"display_name": "不归一化 (None)", "value": "none"}
                ]
            },
            "global_constraints": {
                "label": "启用全局约束:",
                "type": "bool",
                "default": True
            },
            "band_rad": {
                "label": "约束带半径 (0.0-1.0):",
                "type": "float",
                "default": 0.1,
                "validation": {"type": "float", "min": 0.0, "max": 1.0, "decimals": 2}
            }
        }
    },
    "lbp": {
        "display_name": "LBP特征匹配 (LBP)",
        "params": {
            "radius": {
                "label": "LBP采样半径 (Radius):",
                "type": "int",
                "default": 1,
                "validation": {"type": "int", "min": 1, "max": 10}
            },
            "n_points": {
                "label": "LBP采样点数 (Points):",
                "type": "int",
                "default": 8,
                "validation": {"type": "int", "min": 4, "max": 80}
            },
            "method": {
                "label": "LBP提取方法:",
                "type": "dropdown",
                "default": "uniform",
                "options": [
                    {"display_name": "均匀模式 (Uniform)", "value": "uniform"},
                    {"display_name": "非旋转不变均匀 (NRI Uniform)", "value": "nri_uniform"},
                    {"display_name": "原始模式 (Default)", "value": "default"},
                    {"display_name": "旋转不变模式 (ROR)", "value": "ror"}
                ]
            },
            "metric": {
                "label": "LBP直方图比较方法:",
                "type": "dropdown",
                "default": "chi2",
                "options": [
                    {"display_name": "卡方距离 (Chi-Squared)", "value": "chi2"},
                    {"display_name": "交叉核 (Intersection)", "value": "intersection"},
                    {"display_name": "相关性 (Correlation)", "value": "correlation"},
                    {"display_name": "余弦相似度 (Cosine)", "value": "cosine"},
                    {"display_name": "欧氏距离 (Euclidean)", "value": "euclidean"},

                ]
            }
        }
    },
    "ncc": {
        "display_name": "归一化互相关 (NCC)",
        "params": {}
    }
}


