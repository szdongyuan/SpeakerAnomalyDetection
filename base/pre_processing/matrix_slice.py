class MatrixSlice(object):

    @staticmethod
    def matrix_slice(signal, sr, **kwargs):
        """
            preprocess_method: "matrix_slice", which slices a 2D matrix by row and column ranges.

            This is typically used after feature extraction (e.g., spectrogram, MFCC)
            to select a specific region of the feature matrix before AI inference.

            Args:
            - signal: array
                The 2D feature matrix (e.g., spectrogram with shape (time_frames, freq_bins)).
            - sr: int
                The sample rate (unused, kept for interface consistency).
            - **kwargs: Additional parameters
                - row_range: list [start, end]
                    Row slicing range (inclusive start, exclusive end).
                    If not provided, all rows are kept.
                - col_range: list [start, end]
                    Column slicing range (inclusive start, exclusive end).
                    If not provided, all columns are kept.

            Returns:
            - sliced: array
                The sliced sub-matrix.

            Example config:
                preprocess_method: "matrix_slice"
                preprocess_param:
                    row_range: [2, 30]
                    col_range: [6, 70]
        """
        if signal.ndim < 2:
            return signal

        row_range = kwargs.get("row_range")
        col_range = kwargs.get("col_range")

        if row_range:
            signal = signal[row_range[0]:row_range[1]]
        if col_range:
            signal = signal[:, col_range[0]:col_range[1]]

        return signal
