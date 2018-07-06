import json
import codecs

class CharDataProcessor(object):
    def vocab_processor(_, *texts):
        max_document_length = 0
        for text in texts:
            max_doc_len = max([len(line.decode("utf-8")) for line in text])
            if max_doc_len > max_document_length:
                max_document_length = max_doc_len
        return VocabularyProcessor(max_document_length)

    def restore_vocab_processor(_, vocab_path):
        return VocabularyProcessor.restore(vocab_path)

    def clean_data(_, string):
        return string

class VocabularyProcessor(object):
    def __init__(self, max_document_length, min_frequency=0, vocabulary=None,
                       tokenizer_fn=None):
    # init a class. index  maxdocument length and a vocabulabrary
        if vocabulary == None:
            self.vocabulary_ = {"<PAD>":0} # padding
        else:
            self.vocabulary_ = vocabulary

        self.index = 1
        self.max_document_length = max_document_length
    def fit_transform(self, raw_documents, unused_y=None, fit=True):
        result = []
        for raw_document in raw_documents:
            # mark for this, we can find it is a [[I am a  student]]
            result.append([self.__vocab_id(char, fit) for char in raw_document.decode("utf-8")])

        if self.max_document_length == None:
            max_document_length = max([len(vocab_ids) for vocab_ids in result])
        else:
            max_document_length = self.max_document_length

        result = self.__smooth_lengths(result, max_document_length)

        return result

    def transform(self, raw_documents):
        return self.fit_transform(raw_documents, None, False)

    def save(self, file):
        with codecs.open(file, 'w', 'utf-8') as f:
            data = {"vocabulary_": self.vocabulary_, "index": self.index,
                    "max_document_length": self.max_document_length}
            f.write(json.dumps(data, ensure_ascii=False))

    @classmethod
    def restore(cls, file):
        with codecs.open(file, "r", "utf-8") as f:
            data = json.loads(f.readline())
            vp = cls(data["max_document_length"], 0, data["vocabulary_"])
            vp.index = data["index"]
            return vp

    @staticmethod
    def __smooth_lengths(documents, length):
        result = []
        for document in documents:
            if len(document) > length:
                doccument = document[:length]
            elif len(document) < length:
                document = document + [0] * (length - len(document))
            result.append(document)
        return result

    def __vocab_id(self, char, fit = True):
        # every word has a id
        if char not in self.vocabulary_:
            if fit:
                self.vocabulary_[char] = self.index
                self.index += 1
            else:
                char = "<PAD>"
        return self.vocabulary_[char]

