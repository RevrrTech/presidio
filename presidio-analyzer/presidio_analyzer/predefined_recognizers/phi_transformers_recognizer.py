import copy
import logging
import tempfile
import json
from typing import Optional, List

import torch
from robust_deid.ner_datasets import DatasetCreator
from robust_deid.sequence_tagging import SequenceTagger
from robust_deid.sequence_tagging.arguments import (
    ModelArguments,
    DataTrainingArguments,
    EvaluationArguments,
)
from robust_deid.deid import TextDeid

from presidio_analyzer import (
    RecognizerResult,
    EntityRecognizer,
    AnalysisExplanation,
)
from presidio_analyzer.nlp_engine import NlpArtifacts

from presidio_analyzer.predefined_recognizers.config import PHI_TRANSFORMERS_CONFIG

logger = logging.getLogger("presidio-analyzer")

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        pipeline,
        TokenClassificationPipeline,
        TrainingArguments,
        HfArgumentParser,
    )
except ImportError:
    logger.error("transformers is not installed")


class PHITransformersRecognizer(EntityRecognizer):
    """
    Wrapper for a transformers model, if needed to be used within Presidio Analyzer.
    The class loads models hosted on HuggingFace - https://huggingface.co/
    and loads the model and tokenizer into a TokenClassification pipeline.
    Samples are split into short text chunks, ideally shorter than max_length input_ids of the individual model,
    to avoid truncation by the Tokenizer and loss of information

    A configuration object should be maintained for each dataset-model combination and translate
    entities names into a standardized view. Check the PHI_TRANSFORMERS_CONFIG object
    :param supported_entities: List of entities to run inference on
    :type supported_entities: Optional[List[str]]
    :param pipeline: Instance of a TokenClassificationPipeline including a Tokenizer and a Model, defaults to None
    :type pipeline: Optional[TokenClassificationPipeline], optional
    :param model_path: string referencing a HuggingFace uploaded model to be used for Inference, defaults to None
    :type model_path: Optional[str], optional
    """

    def load(self) -> None:
        pass

    def __init__(
        self,
        model_path: Optional[str] = PHI_TRANSFORMERS_CONFIG["DEFAULT_MODEL_PATH"],
        pipeline: Optional[TokenClassificationPipeline] = None,
        supported_entities: Optional[List[str]] = None,
        span_constraint='super_strict',
        sentencizer='en_core_sci_sm',
        tokenizer='clinical',
        max_tokens=128,
        max_prev_sentence_token=32,
        max_next_sentence_token=32,
        default_chunk_size=32,
        ignore_label='NA'
    ):
        if not supported_entities:
            supported_entities = PHI_TRANSFORMERS_CONFIG[
                "PRESIDIO_SUPPORTED_ENTITIES"
            ]
        super().__init__(
            supported_entities=supported_entities,
            name="PHITransformersRecognizer",
        )

        self.model_path = model_path
        self.pipeline = pipeline
        self.is_loaded = False

        self.aggregation_mechanism = None
        self.ignore_labels = None
        self.model_to_presidio_mapping = None
        self.entity_mapping = PHI_TRANSFORMERS_CONFIG["DATASET_TO_PRESIDIO_MAPPING"]
        self.default_explanation = None
        self.text_overlap_length = None
        self.chunk_length = None
        self.id_entity_name = None
        self.id_score_reduction = None
        self._dataset_creator = DatasetCreator(
            sentencizer=sentencizer,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            max_prev_sentence_token=max_prev_sentence_token,
            max_next_sentence_token=max_next_sentence_token,
            default_chunk_size=default_chunk_size,
            ignore_label=ignore_label
        )
        parser = HfArgumentParser((
        ModelArguments,
        DataTrainingArguments,
        EvaluationArguments,
        TrainingArguments
        ))
        model_config = PHITransformersRecognizer._get_model_config()
        model_config['model_name_or_path'] = model_path
        model_config['post_process'] = 'argmax'
        model_config['threshold'] = None
        with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
            tmp.write(json.dumps(model_config) + '\n')
            tmp.seek(0)
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            self._model_args, self._data_args, self._evaluation_args, self._training_args = \
            parser.parse_json_file(json_file=tmp.name)
        
        self._text_deid = TextDeid(notation=model_config['notation'], span_constraint=span_constraint)
        # Initialize the sequence tagger
        self._sequence_tagger = SequenceTagger(
            task_name=self._data_args.task_name,
            notation=self._data_args.notation,
            ner_types=self._data_args.ner_types,
            model_name_or_path=self._model_args.model_name_or_path,
            config_name=self._model_args.config_name,
            tokenizer_name=self._model_args.tokenizer_name,
            post_process=self._model_args.post_process,
            cache_dir=self._model_args.cache_dir,
            model_revision=self._model_args.model_revision,
            use_auth_token=self._model_args.use_auth_token,
            threshold=self._model_args.threshold,
            do_lower_case=self._data_args.do_lower_case,
            fp16=self._training_args.fp16,
            seed=self._training_args.seed,
            local_rank=self._training_args.local_rank
        )
        # Load the required functions of the sequence tagger
        self._sequence_tagger.load()
    
    def get_ner_dataset(self, notes_file):
        ner_notes = self._dataset_creator.create(
            input_file=notes_file,
            mode='predict',
            notation=self._data_args.notation,
            token_text_key='text',
            metadata_key='meta',
            note_id_key='note_id',
            label_key='label',
            span_text_key='spans'
        )
        return ner_notes
    
    def get_predictions(self, ner_notes_file):
        # Set the required data and predictions of the sequence tagger
        # Can also use self._data_args.test_file instead of ner_dataset_file (make sure it matches ner_dataset_file)
        self._sequence_tagger.set_predict(
            test_file=ner_notes_file,
            max_test_samples=self._data_args.max_predict_samples,
            preprocessing_num_workers=self._data_args.preprocessing_num_workers,
            overwrite_cache=self._data_args.overwrite_cache
        )
        # Initialize the huggingface trainer
        self._sequence_tagger.setup_trainer(training_args=self._training_args)
        # Store predictions in the specified file
        predictions = self._sequence_tagger.predict()
        return predictions
    
    @staticmethod
    def _get_model_config():
        return {
            "post_process":None,
            "threshold": None,
            "model_name_or_path":None,
            "task_name":"ner",
            "notation":"BILOU",
            "ner_types":["PATIENT", "STAFF", "AGE", "DATE", "PHONE", "ID", "EMAIL", "PATORG", "LOC", "HOSP", "OTHERPHI"],
            "truncation":True,
            "max_length":512,
            "label_all_tokens":False,
            "return_entity_level_metrics":True,
            "text_column_name":"tokens",
            "label_column_name":"labels",
            "output_dir":"./run/models",
            "logging_dir":"./run/logs",
            "overwrite_output_dir":False,
            "do_train":False,
            "do_eval":False,
            "do_predict":True,
            "report_to":[],
            "per_device_train_batch_size":0,
            "per_device_eval_batch_size":16,
            "logging_steps":1000
        }

    # def load_transformer(self, **kwargs) -> None:
    #     """Load external configuration parameters and set default values.

    #     :param kwargs: define default values for class attributes and modify pipeline behavior
    #     **DATASET_TO_PRESIDIO_MAPPING (dict) - defines mapping entity strings from dataset format to Presidio format
    #     **MODEL_TO_PRESIDIO_MAPPING (dict) -  defines mapping entity strings from chosen model format to Presidio format
    #     **SUB_WORD_AGGREGATION(str) - define how to aggregate sub-word tokens into full words and spans as defined
    #     in HuggingFace https://huggingface.co/transformers/v4.8.0/main_classes/pipelines.html#transformers.TokenClassificationPipeline # noqa
    #     **CHUNK_OVERLAP_SIZE (int) - number of overlapping characters in each text chunk
    #     when splitting a single text into multiple inferences
    #     **CHUNK_SIZE (int) - number of characters in each chunk of text
    #     **LABELS_TO_IGNORE (List(str)) - List of entities to skip evaluation. Defaults to ["O"]
    #     **DEFAULT_EXPLANATION (str) - string format to use for prediction explanations
    #     **ID_ENTITY_NAME (str) - name of the ID entity
    #     **ID_SCORE_REDUCTION (float) - score multiplier for ID entities
    #     """

    #     self.entity_mapping = kwargs.get("DATASET_TO_PRESIDIO_MAPPING", {})
    #     self.model_to_presidio_mapping = kwargs.get("MODEL_TO_PRESIDIO_MAPPING", {})
    #     self.ignore_labels = kwargs.get("LABELS_TO_IGNORE", ["O"])
    #     self.aggregation_mechanism = kwargs.get("SUB_WORD_AGGREGATION", "simple")
    #     self.default_explanation = kwargs.get("DEFAULT_EXPLANATION", None)
    #     self.text_overlap_length = kwargs.get("CHUNK_OVERLAP_SIZE", 40)
    #     self.chunk_length = kwargs.get("CHUNK_SIZE", 600)
    #     self.id_entity_name = kwargs.get("ID_ENTITY_NAME", "ID")
    #     self.id_score_reduction = kwargs.get("ID_SCORE_REDUCTION", 0.5)

    #     if not self.pipeline:
    #         if not self.model_path:
    #             self.model_path = "obi/deid_roberta_i2b2"
    #             logger.warning(
    #                 f"Both 'model' and 'model_path' arguments are None. Using default model_path={self.model_path}"
    #             )

    #     self._load_pipeline()

    # def _load_pipeline(self) -> None:
    #     """Initialize NER transformers pipeline using the model_path provided"""

    #     logging.debug(f"Initializing NER pipeline using {self.model_path} path")
    #     device = 0 if torch.cuda.is_available() else -1
    #     self.pipeline = pipeline(
    #         "ner",
    #         model=AutoModelForTokenClassification.from_pretrained(self.model_path),
    #         tokenizer=AutoTokenizer.from_pretrained(self.model_path),
    #         # Will attempt to group sub-entities to word level
    #         aggregation_strategy=self.aggregation_mechanism,
    #         device=device,
    #         framework="pt",
    #         ignore_labels=self.ignore_labels,
    #     )

    #     self.is_loaded = True

    def get_supported_entities(self) -> List[str]:
        """
        Return supported entities by this model.
        :return: List of the supported entities.
        """
        return self.supported_entities

    # Class to use transformers with Presidio as an external recognizer.
    def analyze(
        self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts = None
    ) -> List[RecognizerResult]:
        """
        Analyze text using transformers model to produce NER tagging.
        :param text : The text for analysis.
        :param entities: Not working properly for this recognizer.
        :param nlp_artifacts: Not used by this recognizer.
        :return: The list of Presidio RecognizerResult constructed from the recognized
            transformers detections.
        """

        results = list()
        # Run transformer model on the provided text
        ner_results = self._get_ner_results_for_text(text)

        for res in ner_results:
            res["entity_group"] = self.__check_label_transformer(res["entity_group"])
            if not res["entity_group"]:
                continue

            if res["entity_group"] == self.id_entity_name:
                print(f"ID entity found, multiplying score by {self.id_score_reduction}")
                res["score"] = res["score"] * self.id_score_reduction

            textual_explanation = self.default_explanation.format(res["entity_group"])
            explanation = self.build_transformers_explanation(
                float(round(res["score"], 2)), textual_explanation, res["word"]
            )
            transformers_result = self._convert_to_recognizer_result(res, explanation)

            results.append(transformers_result)

        return results

    @staticmethod
    def split_text_to_word_chunks(
        input_length: int, chunk_length: int, overlap_length: int
    ) -> List[List]:
        """The function calculates chunks of text with size chunk_length. Each chunk has overlap_length number of
        words to create context and continuity for the model

        :param input_length: Length of input_ids for a given text
        :type input_length: int
        :param chunk_length: Length of each chunk of input_ids.
        Should match the max input length of the transformer model
        :type chunk_length: int
        :param overlap_length: Number of overlapping words in each chunk
        :type overlap_length: int
        :return: List of start and end positions for individual text chunks
        :rtype: List[List]
        """
        if input_length < chunk_length:
            return [[0, input_length]]
        if chunk_length <= overlap_length:
            logger.warning(
                "overlap_length should be shorter than chunk_length, setting overlap_length to by half of chunk_length"
            )
            overlap_length = chunk_length // 2
        return [
            [i, min([i + chunk_length, input_length])]
            for i in range(
                0, input_length - overlap_length, chunk_length - overlap_length
            )
        ]

    def _get_ner_results_for_text(self, text: str) -> List[dict]:
        """The function runs model inference on the provided text.
        The text is split into chunks with n overlapping characters.
        The results are then aggregated and duplications are removed.

        :param text: The text to run inference on
        :type text: str
        :return: List of entity predictions on the word level
        :rtype: List[dict]
        """
        notes = [{"text": text, "meta": {"note_id": "note_1", "patient_id": "patient_1"}, "spans": []}]
        predictions = list()
        # Create temp notes file
        with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
            for note in notes:
                tmp.write(json.dumps(note) + '\n')
            tmp.seek(0)
            ner_notes = self.get_ner_dataset(tmp.name)
        # Create temp ner_notes file    
        with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
            for ner_sentence in ner_notes:
                tmp.write(json.dumps(ner_sentence) + '\n')
            tmp.seek(0)
            predictions = self.get_predictions(tmp.name)

        # model_max_length = self.pipeline.tokenizer.model_max_length
        # # calculate inputs based on the text
        # text_length = len(text)
        # # split text into chunks
        # if text_length <= model_max_length:
        #     predictions = self.pipeline(text)
        # else:
        #     logger.info(
        #         f"splitting the text into chunks, length {text_length} > {model_max_length}"
        #     )
        #     predictions = list()
        #     chunk_indexes = PHITransformersRecognizer.split_text_to_word_chunks(
        #         text_length, self.chunk_length, self.text_overlap_length
        #         )

        #     # iterate over text chunks and run inference
        #     for chunk_start, chunk_end in chunk_indexes:
        #         chunk_text = text[chunk_start:chunk_end]
        #         chunk_preds = self.pipeline(chunk_text)

        #         # align indexes to match the original text - add to each position the value of chunk_start
        #         aligned_predictions = list()
        #         for prediction in chunk_preds:
        #             prediction_tmp = copy.deepcopy(prediction)
        #             prediction_tmp["start"] += chunk_start
        #             prediction_tmp["end"] += chunk_start
        #             aligned_predictions.append(prediction_tmp)

        #         predictions.extend(aligned_predictions)

        # # remove duplicates
        # predictions = [dict(t) for t in {tuple(d.items()) for d in predictions}]
        print("predictions: ", predictions)
        return predictions

    @staticmethod
    def _convert_to_recognizer_result(
        prediction_result: dict, explanation: AnalysisExplanation
    ) -> RecognizerResult:
        """The method parses NER model predictions into a RecognizerResult format to enable down the stream analysis

        :param prediction_result: A single example of entity prediction
        :type prediction_result: dict
        :param explanation: Textual representation of model prediction
        :type explanation: str
        :return: An instance of RecognizerResult which is used to model evaluation calculations
        :rtype: RecognizerResult
        """

        transformers_results = RecognizerResult(
            entity_type=prediction_result["entity_group"],
            start=prediction_result["start"],
            end=prediction_result["end"],
            score=float(round(prediction_result["score"], 2)),
            analysis_explanation=explanation,
        )

        return transformers_results

    def build_transformers_explanation(
        self,
        original_score: float,
        explanation: str,
        pattern: str,
    ) -> AnalysisExplanation:
        """
        Create explanation for why this result was detected.
        :param original_score: Score given by this recognizer
        :param explanation: Explanation string
        :param pattern: Regex pattern used
        :return Structured explanation and scores of a NER model prediction
        :rtype: AnalysisExplanation
        """
        explanation = AnalysisExplanation(
            recognizer=self.__class__.__name__,
            original_score=float(original_score),
            textual_explanation=explanation,
            pattern=pattern,
        )
        return explanation

    def __check_label_transformer(self, label: str) -> Optional[str]:
        """The function validates the predicted label is identified by Presidio
        and maps the string into a Presidio representation
        :param label: Predicted label by the model
        :return: Returns the adjusted entity name
        """

        # convert model label to presidio label
        entity = self.model_to_presidio_mapping.get(label, None)

        if entity in self.ignore_labels:
            return None

        if entity is None:
            logger.warning(f"Found unrecognized label {label}, returning entity as is")
            return label

        if entity not in self.supported_entities:
            logger.warning(f"Found entity {entity} which is not supported by Presidio")
            return entity
        return entity