import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import unittest
import numpy as np
import librosa

from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis


class TestCalculateFundamentalFreq(unittest.TestCase):
    
    def setUp(self):
        # 测试参数
        self.sr = 44100  # 采样率
        self.duration = 1.0  # 信号持续时间(秒)
        self.t = np.linspace(0, self.duration, int(self.sr * self.duration), endpoint=False)
        
        # 生成已知基频的测试信号 - 单一频率的正弦波
        self.fundamental_freq = 440  # A4音符 (440Hz)
        self.reference_signal = np.sin(2 * np.pi * self.fundamental_freq * self.t)
        
        # 添加一些谐波分量的复杂信号
        self.complex_signal = self.reference_signal.copy()
        for harmonic in range(2, 5):
            # 添加谐波，每个谐波的振幅是基频的1/harmonic
            self.complex_signal += (1.0/harmonic) * np.sin(2 * np.pi * harmonic * self.fundamental_freq * self.t)
        
        # 添加一些噪声的信号
        np.random.seed(42)  # 固定随机数种子以确保可重复性
        self.noisy_signal = self.reference_signal + 0.1 * np.random.normal(0, 1, len(self.reference_signal))
    
    def test_calculate_fundamental_freq_yin(self):
        """测试使用YIN算法计算基频"""
        f0, times = AudioThdFrequencyResponseAnalysis.calculate_fundamental_freq(
            self.reference_signal, self.sr, method="yin"
        )
        
        # 检查返回值的类型和形状
        self.assertIsInstance(f0, np.ndarray)
        self.assertIsInstance(times, np.ndarray)
        self.assertEqual(len(f0), len(times))
        
        # 检查计算的基频是否接近预期值
        # 注意：YIN算法可能会有一些误差，所以使用近似比较
        # 我们检查平均基频是否在预期值的5%范围内
        mean_f0 = np.mean(f0)
        self.assertAlmostEqual(mean_f0, self.fundamental_freq, delta=self.fundamental_freq * 0.05)
    
    def test_with_complex_signal(self):
        """测试带有谐波的复杂信号"""
        f0, times = AudioThdFrequencyResponseAnalysis.calculate_fundamental_freq(
            self.complex_signal, self.sr, method="yin"
        )
        
        # 即使有谐波成分，函数仍应该能够找到基频
        mean_f0 = np.mean(f0)
        self.assertAlmostEqual(mean_f0, self.fundamental_freq, delta=self.fundamental_freq * 0.05)
    
    def test_with_noisy_signal(self):
        """测试带有噪声的信号"""
        f0, times = AudioThdFrequencyResponseAnalysis.calculate_fundamental_freq(
            self.noisy_signal, self.sr, method="yin", f0_min=400, f0_max=500
        )
        
        # 在有噪声的情况下，使用频率范围限制可以提高准确性
        mean_f0 = np.mean(f0)
        self.assertAlmostEqual(mean_f0, self.fundamental_freq, delta=self.fundamental_freq * 0.08)


if __name__ == "__main__":
    unittest.main() 