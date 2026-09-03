class SequenceData:
    def __init__(self, seq_name):
        self._seq_name = seq_name
        self._name = None
        self._mode = None
        self._detail = {}
        self._display_sequence = []
        self._data = []
        self._default_ai = None
        self._auto_analysis = True

        self._acq = {"name": self._name, "mode": self._mode, "detail": self._detail}
        self._analysis_list = {
            "display_sequence": self._display_sequence,
            "default_ai": self._default_ai,
            "auto_analysis": self._auto_analysis,
        }

    def _on_display_sequence_change(self, new_value):
        self._analysis_list["display_sequence"] = new_value

    def append(self, item):
        self._data.append(item)
        self._on_display_sequence_change(self._data)

    @property
    def seq_name(self):
        return self._seq_name

    @seq_name.setter
    def seq_name(self, value):
        self._seq_name = value

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, value):
        self._name = value
        self._acq["name"] = value

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value
        self._acq["mode"] = value

    @property
    def detail(self):
        return self._detail

    @detail.setter
    def detail(self, value):
        self._detail = value
        self._acq["detail"] = value

    @property
    def display_sequence(self):
        return self._display_sequence

    @display_sequence.setter
    def display_sequence(self, value):
        self._display_sequence = value
        self._analysis_list["display_sequence"] = value

    @property
    def default_ai(self):
        return self._default_ai

    @default_ai.setter
    def default_ai(self, value):
        self._default_ai = value
        self._analysis_list["default_ai"] = value

    @property
    def auto_analysis(self):
        return self._auto_analysis

    @auto_analysis.setter
    def auto_analysis(self, value):
        self._auto_analysis = value
        self._analysis_list["auto_analysis"] = value

    @property
    def acq(self):
        self._acq["mode"] = self._mode
        self._acq["detail"] = self._detail
        return self._acq

    @property
    def analysis_list(self):
        self._analysis_list["display_sequence"] = self._display_sequence
        self._analysis_list["default_ai"] = self._default_ai
        self._analysis_list["auto_analysis"] = self._auto_analysis
        return self._analysis_list

    @property
    def config_info(self):
        return {self._seq_name: {"acq": self.acq, "analysis_list": self.analysis_list}}
