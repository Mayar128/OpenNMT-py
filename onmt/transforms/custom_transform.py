from onmt.transforms import Transform

class NoOpTransform(Transform):
    def __init__(self, opts):
        super().__init__(opts)
    
    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Pass through without modification since data is pre-processed"""
        return example
    
    @classmethod
    def add_options(cls, parser):
        """No additional options needed"""
        pass