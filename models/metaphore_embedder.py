import logging
import torch
import numpy as np

from overrides import overrides

from allennlp.common import Params, Tqdm
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.elmo import Elmo
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.data import Vocabulary
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)

"""
    Read pre-trained word vectors from a text file.
	The text file is assumed to be utf-8 encoded with
    space-separated fields: [word] [dim 1] [dim 2] ...
    Lines that contain more numerical tokens than ``embedding_dim`` raise a warning and are skipped.
"""
def get_glove_embedder(num_embeddings: int, embedding_dim: int, vocab: Vocabulary, namespace: str = "tokens") -> Embedding:

    tokens_to_keep = set(vocab.get_index_to_token_vocabulary(namespace).values())
    vocab_size = vocab.get_vocab_size(namespace)
    embeddings = {}

    # First we read the embeddings from the file, only keeping vectors for the words we need.
    logger.info("Reading pre-trained embeddings from file")
	
    with open("../embeddings/glove/glove840B300d.txt",encoding="utf8") as embeddings_file:
        for line in Tqdm.tqdm(embeddings_file):
            token = line.split(' ', 1)[0]
            if token in tokens_to_keep:
                fields = line.rstrip().split(' ')
                if len(fields) - 1 != embedding_dim:
                    # Sometimes there are funny unicode parsing problems that lead to different
					# fields lengths (e.g., a word with a unicode space character that splits
                    # into more than one column).  We skip those lines.  Note that if you have
                    # some kind of long header, this could result in all of your lines getting
                    # skipped.  It's hard to check for that here; you just have to look in the
                    # embedding_misses_file and at the model summary to make sure things look
                    # like they are supposed to.
                    logger.warning("Found line with wrong number of dimensions (expected: %d; actual: %d): %s",
					               embedding_dim, len(fields) - 1, line)
                    continue
                
                vector = np.asarray(fields[1:], dtype='float32')
                embeddings[token] = vector
    
    if not embeddings:
        raise ConfigurationError("No embeddings of correct dimension found; you probably "
                                "misspecified your embedding_dim parameter, or didn't "
                                "pre-populate your Vocabulary")

    all_embeddings = np.asarray(list(embeddings.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))
	 
    print("Embedding mean:" + str(embeddings_mean))
    print("Embedding std:" + str(embeddings_std))

    # Now we initialize the weight matrix for an embedding layer, starting with random vectors,
    # then filling in the word vectors we just read.
    logger.info("Initializing pre-trained embedding layer")
    embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(embeddings_mean,embeddings_std)

    num_tokens_found = 0
    index_to_token = vocab.get_index_to_token_vocabulary(namespace)

    for i in range(vocab_size):
        token = index_to_token[i]
        
        # If we don't have a pre-trained vector for this word, we'll just leave this row alone,
        # so the word has a random initialization.
        if token in embeddings:
            embedding_matrix[i] = torch.FloatTensor(embeddings[token])
            num_tokens_found += 1
        else:
            logger.debug("Token %s was not found in the embedding file. Initialising randomly.", token)
    
    logger.info("Pretrained embeddings were found for %d out of %d tokens",
                num_tokens_found, vocab_size)
			
    # initialize glove embedding on precalculated weight
    glove_embedder = Embedding(num_embeddings, embedding_dim, weight = embedding_matrix, padding_index=0)
	
    return glove_embedder
	
@TokenEmbedder.register("metaphore_embedder")
class MetaphoreFieldEmbedder(TokenEmbedder):
    """
    Embedder that is going to be used
	
    Parameters
    ----------
	num_embeddings : int:
    Size of the dictionary of embeddings (vocabulary size + 2 for unknown/padding).

    embedding_dim : int
    The size of total embedding vector
	
    glove embedder : Embedding , Optional - created from glove word vectors (if enabled)
	
    elmo embedder : Embedding , Optional - created from elmo if enabled (https://allennlp.org/elmo)
	
    verb_index_embedder: Embedding , Optional - created from index of verb in the sentence if enabled
	
    visual_score_embedder: Embedding , Optional - created from visual score of word if enabled
"""
    def __init__(self, num_embeddings: int, embedding_dim: int, glove_embedder : Embedding = None, 
                    elmo_embedder : Elmo = None, verb_index_embedder : Embedding = None, 
                    visual_score_embedder : Embedding = None, is_gpu : bool = False):
        super(MetaphoreFieldEmbedder, self).__init__()
        self.num_embeddings = num_embeddings
        self.output_dim = embedding_dim
        self.elmo_embedder = elmo_embedder
        self.glove_embedder = glove_embedder
        self.verb_index_embedder = verb_index_embedder
        self.visual_score_embedder = visual_score_embedder
        self.is_gpu = is_gpu
    
    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim
		
    """
       Gets inputs from data reader and forward it to the network after embedding
    """
    @overrides
    def forward(self, sentences, verb_index, labels): 
        
        if self.glove_embedder != None:		
            # take the result of glove embedding on our batch
            glove_part = self.glove_embedder(sentences["tokens"])
		
        # embed elmo only if using elmo embedding
        if self.elmo_embedder != None:
            elmo_part = self.elmo_embedder(sentences["elmo_characters"])
			
        # embed index of verb
        if self.verb_index_embedder != None:
            # take the result of suffix embedding onindex of verb
            verb_index_part = self.verb_index_embedder(verb_index)

        #embed visual_score	
        if self.visual_score_embedder != None:
            visual_score_part = self.visual_score_embedder(sentences["tokens"])
		
        # get the shape of current input (batch_size,number of words)
        inputShape = sentences["tokens"].shape
        embedded_metaphore = np.array([], dtype=np.float32).reshape(inputShape[0],inputShape[1],0)

        if self.glove_embedder != None:
            embedded_metaphore = np.concatenate((embedded_metaphore, glove_part.cpu().data), axis=2)
        if self.elmo_embedder != None:
            embedded_metaphore = np.concatenate((embedded_metaphore, elmo_part['elmo_representations'][0].cpu().data), axis=2)
        if self.verb_index_embedder != None:
            embedded_metaphore = np.concatenate((embedded_metaphore, verb_index_part.cpu().data), axis=2)
        if self.visual_score_embedder != None:
            embedded_metaphore = np.concatenate((embedded_metaphore, visual_score_part.cpu().data), axis=2)
			
        if embedded_metaphore.shape[2] == 0: 
            raise ConfigurationError("No embedding is used, please enable at least one embedding in configuration.")
	
        if self.is_gpu:
            return torch.cuda.FloatTensor(embedded_metaphore) # This is numpy array. We need to input tensor to network. 
        else:
            return torch.FloatTensor(embedded_metaphore) # This is numpy array. We need to input tensor to network. 
		
    """
        We need the vocabulary here to know how many items we need to embed.
        The format of the glove embedding file if used
        a text file - an utf-8 encoded text file with space separated fields::
        [word] [dim 1] [dim 2] ...
    """
	# Custom logic requires custom from_params.
    @classmethod
    def from_params(cls,params: Params, vocab: Vocabulary) -> 'Embedding':  # type: ignore

        cuda_device = params.pop("cuda_device",-1)
        use_glove_embedding = params.pop("use_glove_embedding", False)
        #glove_dimension_size = params.pop("glove_dimension_size",300)
        use_elmo_embedding = params.pop("use_elmo_embedding", False)
        use_verb_index_embedding = params.pop("use_verb_index_embedding",False)
        verb_index_embedding_dimension = params.pop("verb_index_embedding_dimension",50)
        use_visual_score_embedding = params.pop("use_visual_score_embedding",False)

        num_embeddings = vocab.get_vocab_size() #0 = padding, 1 = unknow, the rest is vocabulary
        embedding_dim = 0
        
        # test if to use elmo embedding 
        if use_elmo_embedding:
            elmo_token_embedder = Elmo.from_params(params.pop("elmo"))
            embedding_dim = embedding_dim + elmo_token_embedder.get_output_dim() # current dimension for elmo embedding - 512*2 = 1024 
        else:
            elmo_token_embedder = None

        if use_glove_embedding:
            # glove_embeddings an Embeddings with dimension of 300
            #glove_embedder = get_glove_embedder(num_embeddings,glove_dimension_size,vocab)
            glove_embedder = Embedding.from_params(vocab, params.pop("glove_embedder"))
            embedding_dim = embedding_dim + glove_embedder.get_output_dim()
        else:
            glove_embedder = None
			
        if use_verb_index_embedding:
            # suffix_embeddings: need two elements for 0 (non-metaphore) and 1 (is metaphore)
            verb_index_embedder = Embedding(2, verb_index_embedding_dimension)
            embedding_dim = embedding_dim + verb_index_embedder.get_output_dim() # for suffix embedding
        else:
            verb_index_embedder = None
			
        if use_visual_score_embedding:
            # use pretrained weight matrix
            visual_score_embedder = Embedding.from_params(vocab, params.pop("visual_embedder"))
            embedding_dim = embedding_dim + visual_score_embedder.get_output_dim()
        else:
            visual_score_embedder = None
			
        if cuda_device == -1:
            is_gpu = False
        else:
            is_gpu = True
	
        return cls(num_embeddings=num_embeddings,embedding_dim=embedding_dim, glove_embedder=glove_embedder, 
                    elmo_embedder=elmo_token_embedder, verb_index_embedder=verb_index_embedder, 
                    visual_score_embedder=visual_score_embedder,is_gpu=is_gpu)