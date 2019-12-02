import logging
import csv
import json
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import WordTokenizer # to split the sentence we read to words
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import SingleIdTokenIndexer,ELMoTokenCharactersIndexer

logger = logging.getLogger(__name__)

@DatasetReader.register("vua_reader")
class VuaDatasetReader(DatasetReader):
    """
    Reads a csv file containing the following columns: {"text_idx", "sentence_idx", "verb", "sentence", "verbIdx", "label"}
    1. text idx in vua corpus
    2. sentence index in vua corus
    3. The verb of the metaphore
    4. Sentence that includes also a pair of verb and an object.
	5. The index of the verb in the sentence.
	6. Label - 1 if the verb-object pair in the sentence is a metaphore, 0 otherwise

    The output of 'read' is a list of ``Instance`` s with the fields:
	1. Sentence of type 'TextField'
	2. VerbIndex of type 'SequenceLabelField'
	3. Label of type 'LabelField'
    """
    def __init__(self) -> None:
        # we use simple word tokenizer to split sentence to words
        self._tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())
		
        #initialize indexers
        singleIdIndexer = SingleIdTokenIndexer()
        elmoIndexer = ELMoTokenCharactersIndexer()
        self.indexers = {}
        self.indexers["tokens"] = singleIdIndexer
        self.indexers["elmo_characters"] = elmoIndexer          
		
    """
    Define how data reader is reaing the input file
    """
    @overrides
    def _read(self, file_path):
	
        instances = []
        
        # count how many sentences were read
        counter = 0
        #open data csv file
        with open(file_path, encoding='latin-1') as f:
            lines = csv.reader(f)

            # raw lines is used to convert csv reader lines to list
            raw_lines = []
            for line in lines:
                counter = counter + 1
                raw_lines.append(line)
		  
            # line[3] is the sentence
            # line[4] is the index of verb in the sentence
            # line[5] is the label - 0 for non-metaphore, 1 for metaphore
            
            for line in raw_lines:
                # Generate tuple of sentence, indexOfVerb, Label for each read operation.
                # This tuple will be used to create the input to train the model using previous embedding
                yield self.text_to_instance(line[3].strip(),line[4], line[5])   
            print('VUA dataset', counter) # print number of lines read from csv file

    # This function will convert the  fields read from csv line to instance.
    # later the trainer will organize instances in a batch

    # Each instance will include: 
    # 1. The sentence for training - TextField. Each token of the sentence will be indexed as
    # Single id for a word and also elmo tocken charecters to be used in case the sentence needs 
    # to be embedded with elmo embedder
    # 2. The index of the verb in a text - SequenceField because we are not indexing this field
    # 3. The label (is metaphore or not) - LabelField
    @overrides
    def text_to_instance(self, sentence: str, verb_index: str = None, label: str = None) -> Instance:

        # tokenize the sentence to words - will be part of the 'text' field in instance
        # use standard database reader tokenizer and indexers
        tokenizedSentence = self._tokenizer.tokenize(sentence)
        # we want to create 1-hot vector of 0/1 for every index of verb
        indicated_sequence = [0] * len(tokenizedSentence)
        indicated_sequence[int(verb_index)] = 1

        # One index is for single indexing of word token.
        # the second will be used for elmo embedder if enabled.
        fields = {'sentences': TextField(tokenizedSentence, self.indexers)}
		
        # send also index of verb - for each word in a sentence, 1 means this is 
        # the verb we are refering when classifying the sentence as metaphore
        if verb_index is not None:
            # sequence of labels. Each label is 0 or 1. It won't be indexed since this is a number.
            # also, each label here referes to one word.
            fields['verb_index'] = SequenceLabelField (indicated_sequence, fields['sentences'])

        if label is not None:
            fields['labels'] = LabelField(int(label),skip_indexing=True)

        return Instance(fields)
