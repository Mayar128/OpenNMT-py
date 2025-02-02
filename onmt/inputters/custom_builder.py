from onmt.inputters.builder import DatasetBuilder

class SignLanguageBuilder(DatasetBuilder):
    def __init__(self, opts):
        super().__init__(opts)
        
    def _get_dataset(self, src_data, tgt_data):
        return SignLanguageDataset(src_data, tgt_data)
        
    def build(self, src_data, tgt_data):
        dataset = self._get_dataset(src_data, tgt_data)
        return dataset