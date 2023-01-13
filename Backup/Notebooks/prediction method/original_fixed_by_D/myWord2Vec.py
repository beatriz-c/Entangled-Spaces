import gensim.models.word2vec as w2v
from timeit import default_timer
from early_stopper import EarlyStopper
from mlflow import log_metric

from gensim.models.word2vec_inner import MAX_WORDS_IN_BATCH


class MyWord2Vec(w2v.Word2Vec):

    def __init__(self,
                 sentences=None,
                 corpus_file=None,
                 vector_size=100,
                 alpha=0.025,
                 window=5,
                 min_count=5,
                 max_vocab_size=None,
                 sample=1e-3,
                 seed=1,
                 workers=3,
                 min_alpha=0.0001,
                 sg=0,
                 hs=0,
                 negative=5,
                 ns_exponent=0.75,
                 cbow_mean=1,
                 hashfxn=hash,
                 epochs=5,
                 null_word=0,
                 trim_rule=None,
                 sorted_vocab=1,
                 batch_words=MAX_WORDS_IN_BATCH,
                 compute_loss=False,
                 callbacks=(),
                 comment=None,
                 max_final_vocab=None,
                 shrink_windows=True,
                 enable_early_stop: bool = True,
                 model_name: str = "experimental1") -> None:
        """Constructor of my word2vec

        Args:
            enable_early_stop (bool, optional): If early stop is enabled. Defaults to True.
            model_name (str, optional): Name to where to save the model in case early stop is enabled. Defaults to "experimental1".
        """
        super().__init__(sentences, corpus_file, vector_size, alpha, window,
                         min_count, max_vocab_size, sample, seed, workers,
                         min_alpha, sg, hs, negative, ns_exponent, cbow_mean,
                         hashfxn, epochs, null_word, trim_rule, sorted_vocab,
                         batch_words, compute_loss, callbacks, comment,
                         max_final_vocab, shrink_windows)
        self.early_stopper = EarlyStopper(
            10, True, 0.0,
            saved_model_name=model_name) if enable_early_stop else None

    def train(self,
              corpus_iterable=None,
              corpus_file=None,
              total_examples=None,
              total_words=None,
              epochs=None,
              start_alpha=None,
              end_alpha=None,
              word_count=0,
              queue_factor=2,
              report_delay=1,
              compute_loss=False,
              callbacks=...,
              **kwargs):

        self.alpha = start_alpha or self.alpha
        self.min_alpha = end_alpha or self.min_alpha
        self.epochs = epochs

        self._check_training_sanity(epochs=epochs,
                                    total_examples=total_examples,
                                    total_words=total_words)
        self._check_corpus_sanity(corpus_iterable=corpus_iterable,
                                  corpus_file=corpus_file,
                                  passes=epochs)

        self.add_lifecycle_event(
            "train",
            msg=
            (f"training model with {self.workers} workers on {len(self.wv)} vocabulary and "
             f"{self.layer1_size} features, using sg={self.sg} hs={self.hs} sample={self.sample} "
             f"negative={self.negative} window={self.window} shrink_windows={self.shrink_windows}"
             ),
        )

        self.compute_loss = compute_loss
        self.running_training_loss = 0.0

        for callback in callbacks:
            callback.on_train_begin(self)

        trained_word_count = 0
        raw_word_count = 0
        start = default_timer() - 0.00001
        job_tally = 0

        for cur_epoch in range(self.epochs):
            for callback in callbacks:
                callback.on_epoch_begin(self)

            if corpus_iterable is not None:
                trained_word_count_epoch, raw_word_count_epoch, job_tally_epoch = self._train_epoch(
                    corpus_iterable,
                    cur_epoch=cur_epoch,
                    total_examples=total_examples,
                    total_words=total_words,
                    queue_factor=queue_factor,
                    report_delay=report_delay,
                    callbacks=callbacks,
                    **kwargs)
            else:
                trained_word_count_epoch, raw_word_count_epoch, job_tally_epoch = self._train_epoch_corpusfile(
                    corpus_file,
                    cur_epoch=cur_epoch,
                    total_examples=total_examples,
                    total_words=total_words,
                    callbacks=callbacks,
                    **kwargs)

            trained_word_count += trained_word_count_epoch
            raw_word_count += raw_word_count_epoch
            job_tally += job_tally_epoch

            for callback in callbacks:
                callback.on_epoch_end(self)

            loss = self.get_latest_training_loss()
            log_metric('Loss', loss, step=cur_epoch)
            if self.early_stopper:
                self.early_stopper(loss, self)
                if self.early_stopper.early_stop:
                    break

        # Log overall time
        total_elapsed = default_timer() - start
        self._log_train_end(raw_word_count, trained_word_count, total_elapsed,
                            job_tally)

        self.train_count += 1  # number of times train() has been called
        self._clear_post_train()

        for callback in callbacks:
            callback.on_train_end(self)

        return trained_word_count, raw_word_count