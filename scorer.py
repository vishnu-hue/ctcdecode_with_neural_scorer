from ctcdecode._ext import ctc_decode
from ctcdecode import ctcdecode
class CTCWithGPT2LM(ctcdecode.CTCBeamDecoder):
    def __init__(self, labels, model_path=None, alpha=0, beta=0, cutoff_top_n=40, cutoff_prob=1.0, beam_width=100,
                 num_processes=4, blank_id=0, log_probs_input=False, max_order = 3, vocab_path = None, have_dictionary = True):
        self.cutoff_top_n = cutoff_top_n
        self._beam_width = beam_width
        self._scorer = None
        self._num_processes = num_processes
        self._labels = list(labels)  # Ensure labels are a list
        self._num_labels = len(labels)
        self._blank_id = blank_id
        self._log_probs = 1 if log_probs_input else 0
        if model_path:
            self._scorer = self.get_scorer(alpha, beta, model_path.encode(), self._labels, max_order, vocab_path.encode(), have_dictionary)
        self._cutoff_prob = cutoff_prob

    def get_scorer(self, alpha, beta, model_path, labels, max_order, vocab_path, have_dictionary):
        return ctc_decode.paddle_get_scorer(alpha, beta, model_path, labels, max_order, vocab_path, have_dictionary)
    
    def release_scorer(self):
        if self._scorer is not None:
            ctc_decode.paddle_release_scorer(self._scorer)
