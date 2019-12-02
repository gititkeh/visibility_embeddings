from typing import Dict, Optional

import numpy
import torch
import torch.nn as nn
from torch.nn import Dropout
from overrides import overrides

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules import Seq2SeqEncoder, FeedForward
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.nn import util
from allennlp.nn.util import masked_log_softmax

from models.metaphore_embedder import MetaphoreFieldEmbedder


@Model.register("metaphore_classifier")
class MetaphoreClassifier(Model):

    """
	This `model performs text classification for a sentence + index of verb
	in the sentence. 
	
    We assume we're given tokens created from the sentence and the index of the verb
	in each sentence, then we predict some output label.
    
    The basic model structure: 
    1. we'll embed each token of the sentence and the index of verb
    2. Apply dropout
    3. Run LSTM with dropout
    4. Run attention of softmax classifier on LSTM output
    5. Apply dropout
    6. Run softmax classifier

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required for base model and embedder
	model_sentence_field_embedder : ``MetaphoreFieldEmbedder``, required
        Used to embed the input to the model.
    input_encoder_dropout: Dropout, required
        Used as a dropout to LSTM
    internal_sentence_encoder: Seq2SeqEncoder, required
        the LSTM we run with embedding to train target
    linear_attention_feedforward: FeedForward, required
        Linear layer used for calculating logits for attention
    input_classifier_dropout: Dropout, required
        Used as dropout to softmax classifier
    linear_classifier_feedforward: FeedForward, required
        Linear layer used for calculating logits for classification
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 model_sentence_field_embedder: MetaphoreFieldEmbedder,
                 input_encoder_dropout : Dropout,
                 internal_sentence_encoder: Seq2SeqEncoder,
                 linear_attention_feedforward : FeedForward,
                 input_classifier_dropout: Dropout,
                 linear_classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        # initialize model
        super(MetaphoreClassifier, self).__init__(vocab, regularizer)
        initializer(self)
        
        self.model_sentence_field_embedder = model_sentence_field_embedder
        self.input_encoder_dropout = input_encoder_dropout
        self.internal_sentence_encoder = internal_sentence_encoder
        self.linear_attention_feedforward = linear_attention_feedforward
        self.input_classifier_dropout = input_classifier_dropout
        self.linear_classifier_feedforward = linear_classifier_feedforward
		
        # we measure for metrics f score of each class separately
        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "_f_score_literal":  F1Measure(0), # positive label index for literal is 0
                "_f_score_metaphore":  F1Measure(1) # positive label index for metaphore is 1
        }
		
        # the loss criterion used as input to trainer optimizer
        self.nll_criterion = nn.NLLLoss()

    @overrides
    def forward(self,
                sentences: Dict[str, torch.LongTensor],
				verb_index: Dict[str, torch.LongTensor] = None,
                labels: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        sentences : Dict[str, torch.LongTensor], required
                This is the list of tokens of sentence in one instance indexed as single id and as elmo embedder input
        verb_index: Dict[str, torch.LongTensor], optional
		    For each word in the sentence, the value is 0 if the word is the metaphore verb in the sentence 
        labels : torch.LongTensor, optional (default = None)
            A variable representing the label for each instance in the batch.
        Returns
        -------
        An output dictionary containing the loss calculated from the forward pass
        """

        # apply the embedder on the text field. the result should be a tensor with the shape
        # (batch_size,number_of_words,embedding_dimension)
        embedded_sentences = self.model_sentence_field_embedder(sentences,verb_index,labels)
		
        # Apply droupout to the output of embedding and before feeding LSTM layer
        input_to_lstm = self.input_encoder_dropout(embedded_sentences)
	
        # find the mask of non-padding elements for the text field
        # we need to do it since the LSTM layer needs ro process only non-padded elements
        # we find the mask in non-embedded sentences
        sentence_mask = util.get_text_field_mask(sentences)
        
        # apply LSTM to embedded text after dropout
        encoded_sentence = self.internal_sentence_encoder(input_to_lstm, sentence_mask)
        
        # Calculae score logits calculated by running linear layer on LSTM result
        attention_logits = self.linear_attention_feedforward(encoded_sentence).squeeze(-1)
  
        # create a mask to calculate attention only on non-0 logits
        attention_logits_mask = (attention_logits != 0)
  
        # Calculate softmax log on the logits
        softmax_attention_logits = masked_log_softmax(attention_logits, attention_logits_mask)
		
        # Unsqueeze vector dimension to match attention with matrix
        softmax_attention_logits = softmax_attention_logits.unsqueeze(dim=1)
		
        # Calculate attention of linear output vector and lstm output matrix
        attention_vector_matrix = torch.bmm(softmax_attention_logits, encoded_sentence)
		
        # squeeze again the attention to match one result per output dimension
        attention_vector_matrix = attention_vector_matrix.squeeze(dim=1)
		
        # Apply droupout to the output of attention layer and input to linear proection
        input_to_classifier = self.input_classifier_dropout(attention_vector_matrix)
		
        #Run last linear layer to transform logits to classifications
        classifier_logits = self.linear_classifier_feedforward(input_to_classifier)
		
        # Normalize with log softmax
        classified_output = masked_log_softmax(classifier_logits, mask=None, dim=-1)
		
        output_dict = {}
		
        # we can compute loss only if we have label to compare with
        if labels is not None:
            loss = self.nll_criterion(classified_output, labels)
            for metric in self.metrics.values():
                metric(classified_output, labels)
            output_dict["loss"] = loss

        return output_dict

    """
    Calculate metrics of training.
    We measure the accuracy and f score for each label class (0 - literal, 1 - mtaphore)
    """
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        
        # initialize
        total_precision = 0.0
        total_recall = 0.0
        total_f_score= 0.0
        result = {}
		
        # handle each metrics
        for metric_name, metric in self.metrics.items():
            
            # literal f score
            if metric_name == '_f_score_literal':
                precision, recall, f1_measure = metric.get_metric(reset)
                result["precision_literal"] = precision
                result["recall_literal"] = recall
                result["f1_literal"] = f1_measure
                total_precision = total_precision + precision
                total_recall = total_recall + recall
                total_f_score = total_f_score + f1_measure
            
            # metaphore f score 
            elif metric_name == '_f_score_metaphore':
                precision, recall, f1_measure = metric.get_metric(reset)
                result["precision_metaphore"] = precision
                result["recall_metaphore"] = recall
                result["f1_metaphore"] = f1_measure
                total_precision = total_precision + precision
                total_recall = total_recall + recall
                total_f_score = total_f_score + f1_measure
            
            # accuracy 
            else:
                result[metric_name] = metric.get_metric(reset)
				
        # calculating average f score
        result["avg_precision"] = total_precision / 2
        result["avg_recall"] = total_recall / 2
        result["avg_f1"] = total_f_score / 2
        return result

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'MetaphoreClassifier':
        embedder_params = params.pop("model_sentence_field_embedder")
        model_sentence_field_embedder = MetaphoreFieldEmbedder.from_params(embedder_params, vocab=vocab)
        input_encoder_dropout = Dropout(params.pop("input_encoder_dropout"))
        internal_sentence_encoder = Seq2SeqEncoder.from_params(params.pop("internal_sentence_encoder"))
        linear_attention_feedforward = FeedForward.from_params(params.pop("linear_attention_feedforward"))
        input_classifier_dropout = Dropout(params.pop("input_classifier_dropout"))
        linear_classifier_feedforward = FeedForward.from_params(params.pop("linear_classifier_feedforward"))
        return cls(vocab=vocab,
                   model_sentence_field_embedder=model_sentence_field_embedder,
                   input_encoder_dropout = input_encoder_dropout,
                   internal_sentence_encoder=internal_sentence_encoder,
                   linear_attention_feedforward = linear_attention_feedforward,
                   input_classifier_dropout = input_classifier_dropout,
                   linear_classifier_feedforward=linear_classifier_feedforward,)
