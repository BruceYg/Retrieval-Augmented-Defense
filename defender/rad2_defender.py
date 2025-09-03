import sys
import os
import csv
import ast
sys.path.append('./src')
import llms
import encoders
import faiss
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from defender import BaseDefender


class Reranker:
    def __init__(self,
                 llm,
                 ) -> None:
        self.llm = llm
        # self.llm = getattr(llms, llm_class)(**llm_config)

    def rerank(self, prompt, contexts, compare=False, likelihood=False, relative_llh=True, length_norm=False):
        if isinstance(contexts, str):
            mode = 'single'
            context = contexts
        else:
            mode = 'batch'
            if isinstance(prompt, str):
                prompts = [prompt] * len(contexts)
            else:
                assert len(prompt) == len(contexts)
                prompts = prompt
        if mode == 'single':
            log_prob = self.llm.calculate_conditional_prob(
                prompt=prompt,
                context=context,
                length_norm=length_norm
            )
            return log_prob
        else:
            log_probs = self.llm.calculate_conditional_prob_batch(
                prompts=prompts,
                contexts=contexts,
                length_norm=length_norm
            )
            if relative_llh:
                base_log_prob = log_probs[-1]
                log_probs = [log_prob - base_log_prob for log_prob in log_probs]
            return log_probs


class Analyzer:
    # NOTE: LEGACY
    def __init__(self,
                 llm_class: str,
                 llm_config: dict,
                 ) -> None:
        if llm_class == 'OpenaiLLM':
            self.mode = 'sequential'
        else:
            self.mode = 'parallel'
        self.llm = getattr(llms, llm_class)(**llm_config)
        self.analyzer_prompt =  open('/home/gy266/rds/hpc-work/RBD/src/prompts/analyzer_prompt.txt', 'r').read()

    def analyze(self, examples):
        analysis_list = []
        guidance_list = []
        if self.mode == 'sequential':
            for example in examples:
                prompt = self.analyzer_prompt.format(prompt=example[0], intent=example[1])
                response = self.llm.generate_response(prompt)
                try:
                    guidance = response.split('Guidance:')[1].strip()
                except:
                    guidance = response.split('\n')[-1].strip()
                try:
                    analysis = response.split('Guidance:')[0].strip().split('Analysis:')[1].strip()
                except:
                    analysis = response.split('\n')[0].strip()
                guidance_list.append(guidance)
                analysis_list.append(analysis)
        else:
            prompts = [self.analyzer_prompt.format(prompt=example[0], intent=example[1]) for example in examples]
            responses = self.llm.generate_response(prompts)
            for response in responses:
                guidance = response.split('Guidance:')[1].strip()
                analysis = response.split('Guidance:')[0].strip().split('Analysis:')[1].strip()
                guidance_list.append(guidance)
                analysis_list.append(analysis)
        return analysis_list, guidance_list


class Extractor:
    def __init__(self,
                 llm,
                 guide: bool = False,
                 ) -> None:
        self.llm = llm
        self.extractor_sg_prompt = open('/home/gy266/rds/hpc-work/RBD/src/prompts/extractor_rag_prompt_reason_analyzed.txt', 'r').read()
        self.extractor_sgpi_prompt = open('/home/gy266/rds/hpc-work/RBD/src/prompts/extractor_rag_prompt_reason_analyzed_v17.txt', 'r').read()

    def postprocess_intent(self, intent):
        if '</think>' in intent:
            intent = intent.split('</think>')[-1].strip()
        if intent.startswith('Extracted Intent:'):
            intent = intent.split('Extracted Intent:')[1].strip()
        if intent.startswith('Intent:'):
            intent = intent.split('Intent:')[1].strip()
        if 'Summary:' in intent:
            intent = intent.split('Summary:')[0].strip()
        if "Final Intent" in intent:
            intent = intent.split("Final Intent")[1].strip()
        if "Extracted Intent: " in intent:
            intent = intent.split("Extracted Intent: ")[1].strip()
        if "Extracted Intent" in intent:
            intent = intent.split("Extracted Intent")[1].strip()
        return intent

    def extract(self, prompt, examples=None, prompt_v18=False, return_selection=False):
        sys_prompt = "You are an Intent Extraction Assistant specialized in accurately identifying and extracting user intents from prompts. Your goal is to carefully analyze the provided strategies, clearly articulate your reasoning, and explicitly state the user's actual intent."
        if len(examples[0]) == 2:
            examples = [f'Strategy: {example[0]}\nGuidance: {example[1]}' for i, example in enumerate(examples)]
            full_prompts = [self.extractor_sg_prompt.format(prompt=prompt, examples=example) for example in examples]
        elif len(examples[0]) == 4:
            examples = examples = [f'Strategy: {example[0]}\nGuidance: {example[1]}\nPrompt: {example[2]}\nIntent: {example[3]}' for i, example in enumerate(examples)]
            full_prompts = [self.extractor_sgpi_prompt .format(prompt=prompt, examples=example) for example in examples]
        print(f'****User prompt: {prompt}')
        extracted_intents = self.llm.generate_response_batch(full_prompts, sys_prompt=sys_prompt)
        [print(f'****Extracted intent {i+1}: {extracted_intent}') for i, extracted_intent in enumerate(extracted_intents)]
        selection_raw = []
        for extracted_intent in extracted_intents:
            try:
                sel_raw = extracted_intent.split('Strategy Selection:')[1].strip()
            except:
                sel_raw = extracted_intent
            selection_raw.append(sel_raw)
        # selection_raw = [extracted_intent.split('Strategy Selection:')[1].strip() for extracted_intent in extracted_intents]
        selection = []
        for sel in selection_raw:
            try:
                selection.append(sel.split('Intent Extraction Process')[0].split('. ')[0])
            except:
                selection.append(sel)
        if prompt_v18:
            # try:
            #     selection = selection.split('Intent Extraction Process')[0]
            # except:
            #     selection = selection
            print(f'****Selection: {selection}')
            # full_prompts = [f'Evaluate whether the following sentences state that at least one of the provided strategies aligns with the user prompt.\nIf at least one strategy aligns, output "Yes". Otherwise, output "No".\nSentences:\n{sel}\nOutput:' for sel in selection]
            selection_template = open('/home/gy266/rds/hpc-work/RBD/src/prompts/intent_selection_prompt.txt', 'r').read()
            full_prompts = [selection_template.format(sel=sel) for sel in selection]
            selection_raw = self.llm.generate_response_batch(full_prompts)
            print(f'****Selection raw: {selection_raw}')
            selection = []
            for sel in selection_raw:
                if sel.startswith("Yes"):
                    selection.append(1)
                elif sel.startswith("No"):
                    selection.append(0)
                else:
                    selection.append(1)
        intents = [self.postprocess_intent(extracted_intent) for extracted_intent in extracted_intents]
        if return_selection:
            return intents, selection
        else:
            return intents
    
    def choose_intent(self, prompt, examples):
        examples = [f'Analysis {i+1}: {example[0]}\nHypothesis {i+1}: {example[1]}' for i, example in enumerate(examples)]
        examples = '\n\n'.join(['- ' + example for example in examples])
        full_prompt = self.extractor_choose_intent_prompt.format(prompt=prompt, examples=examples)
        response = self.llm.generate_response(full_prompt)
        print(f'****Choose intent response: {response}')
        return self.postprocess_intent(response)


class Classifier:
    def __init__(self,
                 llm,
                 guide: bool = False,
                 ) -> None:
        self.llm = llm
        self.classifier_prompt = open('/home/gy266/rds/hpc-work/RBD/src/prompts/classifier_prompt.txt', 'r').read()
        self.classifier_rag_prompt = open('/home/gy266/rds/hpc-work/RBD/src/prompts/classifier_rag_prompt_logit.txt', 'r').read()
        self.classifier_rag_nostrategy_prompt = open('/home/gy266/rds/hpc-work/RBD/src/prompts/classifier_rag_prompt_logit_noStrategy.txt', 'r').read()
        self.classifier_rag_intentonly_prompt = open('/home/gy266/rds/hpc-work/RBD/src/prompts/classifier_rag_prompt_logit_intentOnly.txt', 'r').read()
        self.classifier_rag_prompt_logprob = open('/home/gy266/rds/hpc-work/RBD/src/prompts/classifier_rag_prompt_logprob.txt', 'r').read()

    def _parser(self, response):
        # Find the line containing "Classification:"
        for line in response.lower().split('\n'):
            if "classification:" in line:
                # Get the first word after "Classification:"
                classification = line.split("classification:")[1].strip('[].,!?*').split()[0].strip('[].,!?*')
                if classification == "harmful":
                    return 1
                elif classification == "benign":
                    return 0
                else:
                    print(f'****Invalid classification response: {response}\nTreating as benign.')
                    return 0
        
        print(f'****Invalid classification response: {response}\nTreating as benign.')
        return 0

    def classify(self, prompt, intent=None, strategy=None, output_logits=False, output_logprob=False, intent_only=False):
        print(f'****Classifier input: {intent}')
        if output_logits:
            if isinstance(intent, list):
                # For batch processing, use the batch method if available, otherwise fall back to individual calls
                if hasattr(self.llm, 'classify_first_token_batch'):
                    if intent_only:
                        full_prompts = [self.classifier_rag_intentonly_prompt.format(intent=intent_i) for intent_i in intent]
                    elif strategy is None:
                        full_prompts = [self.classifier_rag_nostrategy_prompt.format(prompt=prompt, intent=intent_i) for intent_i in intent]
                    else:
                        full_prompts = [self.classifier_rag_prompt.format(prompt=prompt, intent=intent_i, strategy=strategy_i) for intent_i, strategy_i in zip(intent, strategy)]
                    yes_probs, no_probs = self.llm.classify_first_token_batch(full_prompts, log_softmax=output_logprob)
                    response = "Classification completed"
                else:
                    # Fallback to individual calls using single-prompt method
                    yes_probs = []
                    no_probs = []
                    for intent_i in intent:
                        if intent_only:
                            full_prompt = self.classifier_rag_intentonly_prompt.format(intent=intent_i)
                        elif strategy is None:
                            full_prompt = self.classifier_rag_nostrategy_prompt.format(prompt=prompt, intent=intent_i)
                        else:
                            # Find corresponding strategy
                            strategy_i = strategy[intent.index(intent_i)] if strategy else None
                            full_prompt = self.classifier_rag_prompt.format(prompt=prompt, intent=intent_i, strategy=strategy_i)
                        
                        if hasattr(self.llm, 'classify_first_token'):
                            yes_prob, no_prob = self.llm.classify_first_token(full_prompt, log_softmax=output_logprob)
                        else:
                            # Fallback to original method
                            response, yes_prob, no_prob = self.llm.generate_response(full_prompt, output_logits=True, output_cls_logits=True)
                        yes_probs.append(yes_prob)
                        no_probs.append(no_prob)
                    response = "Classification completed"
            else:
                # Single prompt case
                if intent_only:
                    full_prompt = self.classifier_rag_intentonly_prompt.format(intent=intent)
                elif strategy is None:
                    full_prompt = self.classifier_rag_nostrategy_prompt.format(prompt=prompt, intent=intent)
                else:
                    full_prompt = self.classifier_rag_prompt.format(prompt=prompt, intent=intent, strategy=strategy)
                
                # Use single-prompt method for efficiency
                if hasattr(self.llm, 'classify_first_token'):
                    yes_prob, no_prob = self.llm.classify_first_token(full_prompt, log_softmax=output_logprob)
                    response = "Classification completed"
                else:
                    # Fallback to original method
                    response, yes_prob, no_prob = self.llm.generate_response(full_prompt, output_logits=True, output_cls_logits=True)
                    try:
                        response = response.split('\n')[0].strip()
                    except:
                        response = response
            print(f'****Classifier response: {response}')
            print(f'****Yes probability: {yes_prob if not isinstance(intent, list) else yes_probs}')
            print(f'****No probability: {no_prob if not isinstance(intent, list) else no_probs}')
            return (yes_prob, no_prob) if not isinstance(intent, list) else (yes_probs, no_probs)
        elif output_logprob:
            full_prompt = self.classifier_rag_prompt_logprob.format(prompt=prompt, intent=intent, strategy=strategy)
            if hasattr(self.llm, 'classify_first_token'):
                yes_prob, no_prob = self.llm.classify_first_token(full_prompt, log_softmax=True)
                response = "Classification completed"
            else:
                # Fallback to original method
                response, yes_prob, no_prob = self.llm.generate_response(full_prompt, output_logits=True, output_cls_logits=True, log_softmax=True)
            print(f'****Classifier response: {response}')
            print(f'****Yes probability: {yes_prob}')
            print(f'****No probability: {no_prob}')
            return yes_prob, no_prob
        else:
            full_prompt = self.classifier_prompt.format(intent=prompt)
            response = self.llm.generate_response(full_prompt)
            print(f'****Classifier response: {response}')
            return self._parser(response)


class RAD2Defender(BaseDefender):
    """
    Retrieval-augmented defense v2.
    """
    def _safe_list_eval(self, value):
        if pd.isna(value) or value == '[]' or value == '':
            return []
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []

    def __init__(self,
                 defender_llm_class: str,
                 defender_llm_config: dict,
                 parse_reasoning: bool = False,
                 analyzer_llm_class: str = None,
                 analyzer_llm_config: dict = None,
                 reranker_llm_class: str = None,
                 reranker_llm_config: dict = None,
                 reranker_likelihood: bool = False,
                 reranker_paraphrase: bool = False,
                 reranker_margin: bool = False,
                 extractor_llm_class: str = None,
                 extractor_llm_config: dict = None,
                 extractor_guide: bool = False,
                 intent_rerank: bool = False,
                 return_intent: bool = True,
                 classifier_llm_class: str = None,
                 classifier_llm_config: dict = None,
                 classifier_guide: bool = False,
                 two_way_retrieve: bool = False,
                 retriever_class_list: list[str] = [None, None],
                 retriever_config_list: list[dict] = [None, None],
                 k: list[int] = [5, 5],
                 run_index: list[bool] = [False, False],
                 jailbreak_db_index_file: str = None,
                 jailbreak_db_source_file: str = None,
                 query_db_index_file: str = None,
                 query_db_source_file: str = None,
                 as_user_prompt: bool = False,
                 test: bool = False,
                 test_2: bool = False,
                 test_3: bool = False,
                 test_test: bool = False,
                 clean: bool = False,
                 clean_norm: bool = False,
                 output_dir: str = None,
                 output_file_name: str = None,
                 uni_posterior: bool = False,
                 output_label: int = 1,
                 no_strategy_classify: bool = False,
                 no_pmi: bool = False,
                 strategy_rerank: bool = False,
                 no_rerank: bool = False,
                 intent_only_classify: bool = False,
                 paraphrase_no_strategy: bool = False,
                 new_prompt: bool = False,
                 cls_threshold: float = 0.5,
                 precompute_file: str = None,
                 precompute_response_file: str = None,
                ) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = getattr(llms, defender_llm_class)(**defender_llm_config)
        if defender_llm_class == 'OpenaiLLM' or defender_llm_config['model_path'] != 'Qwen/Qwen2.5-14B-Instruct':
            self.agent_llm = getattr(llms, extractor_llm_class)(**extractor_llm_config)
        self.parse_reasoning = parse_reasoning
        self.as_user_prompt = as_user_prompt
        self.test = test
        self.test_2 = test_2
        self.test_3 = test_3
        self.test_test = test_test
        self.clean = clean
        self.clean_norm = clean_norm
        self.uni_posterior = uni_posterior
        self.intent_rerank = intent_rerank
        self.counter = 0
        self.output_label = output_label
        self.analyzer = None
        self.cls_threshold = cls_threshold
        self.no_strategy_classify = no_strategy_classify
        self.no_pmi = no_pmi
        self.strategy_rerank = strategy_rerank
        self.no_rerank = no_rerank
        self.intent_only_classify = intent_only_classify
        self.paraphrase_no_strategy = paraphrase_no_strategy
        self.new_prompt = new_prompt
        if precompute_file is not None:
            self.precompute_df = pd.read_csv(precompute_file)
        else:
            self.precompute_df = None
        if precompute_response_file is not None:
            self.precompute_response_df = pd.read_csv(precompute_response_file)
        else:
            self.precompute_response_df = None
        if analyzer_llm_class is not None and analyzer_llm_config is not None:
            self.analyzer = Analyzer(analyzer_llm_class, analyzer_llm_config)
        if output_dir is not None and output_file_name is not None:
            self.scores_output_path = Path(os.path.join(output_dir, output_file_name+"_scores"))
            self.scores_output_path = self.scores_output_path.with_suffix(".csv")
            
            with open(self.scores_output_path, 'w') as f:
                f.write('request_id,prompt,pmi_hypotheses,intent_posterior_hypotheses,softmax_hypotheses,classifier_probs,label\n')
        else:
            self.scores_output_path = None
        
        self.reranker = None
        if reranker_llm_class is not None and reranker_llm_config is not None:
            if defender_llm_class == 'OpenaiLLM' or defender_llm_config['model_path'] != 'Qwen/Qwen2.5-14B-Instruct':
                self.reranker = Reranker(self.agent_llm)
            else:
                self.reranker = Reranker(self.llm)
            # self.reranker = Reranker(reranker_llm_class, reranker_llm_config)
            self.reranker_likelihood = reranker_likelihood
            self.reranker_paraphrase = reranker_paraphrase
            self.reranker_margin = reranker_margin
        self.extractor = None
        self.return_intent = return_intent
        if extractor_llm_class is not None and extractor_llm_config is not None:
            if defender_llm_class == 'OpenaiLLM' or defender_llm_config['model_path'] != 'Qwen/Qwen2.5-14B-Instruct':
                self.extractor = Extractor(self.agent_llm, extractor_guide)
            else:
                self.extractor = Extractor(self.llm, extractor_guide)
            self.defender_intent_sys_prompt = open('/home/gy266/rds/hpc-work/RBD/src/prompts/defender_intent_sys_prompt.txt', 'r').read()
            self.defender_intent_sys_rag_prompt = open('/home/gy266/rds/hpc-work/RBD/src/prompts/defender_intent_sys_rag_prompt.txt', 'r').read()
            if self.as_user_prompt:
                self.defender_intent_user_prompt = open('/home/gy266/rds/hpc-work/RBD/src/prompts/defender_intent_user_prompt.txt', 'r').read()
                self.defender_intent_user_rag_prompt = open('/home/gy266/rds/hpc-work/RBD/src/prompts/defender_intent_user_rag_prompt.txt', 'r').read()
            # self.return_intent = return_intent
        # self.defender_intent_prompt = None

        self.classifier = None
        if classifier_llm_class is not None and classifier_llm_config is not None:
            if defender_llm_class == 'OpenaiLLM' or defender_llm_config['model_path'] != 'Qwen/Qwen2.5-14B-Instruct':
                self.classifier = Classifier(self.agent_llm, classifier_guide)
            else:
                self.classifier = Classifier(self.llm, classifier_guide)
            # self.classifier = Classifier(classifier_llm_class, classifier_llm_config, classifier_guide)
        self.retrievers = []
        for retriever_class, retriever_config in zip(retriever_class_list, retriever_config_list):
            if retriever_class is not None and retriever_config is not None:
                self.retrievers.append(getattr(encoders, retriever_class)(**retriever_config))
            else:
                self.retrievers.append(None)
        self.k = k
        self.two_way_retrieve = two_way_retrieve
        if self.two_way_retrieve:
            assert len(self.retrievers) == 2
            assert self.retrievers[0] is not None and self.retrievers[1] is not None

        if jailbreak_db_index_file is not None and run_index[0] is False:
            assert jailbreak_db_source_file is not None
            print(f'Loading jailbreak database from {jailbreak_db_index_file}...')
            self.jailbreak_db = faiss.read_index(jailbreak_db_index_file)
            jailbreak_db_source_df = pd.read_csv(jailbreak_db_source_file)
            self.jailbreak_db_source = []
            self.jailbreak_db_seed_queries = []
            self.jailbreak_db_scheme = []
            self.jailbreak_db_analysis = []
            self.jailbreak_db_red_flags = []
            self.jailbreak_db_guidance = []
            for _, row in jailbreak_db_source_df.iterrows():
                self.jailbreak_db_source.append(row['jailbreak'])
                self.jailbreak_db_seed_queries.append(row['query'])
                if 'scheme' in row:
                    self.jailbreak_db_scheme.append(row['scheme'])
                if 'analysis' in row:
                    row_analysis = row['analysis']
                    try:
                        row_analysis = row_analysis.split('Analysis: ')[1].strip()
                    except:
                        row_analysis = row_analysis
                    # try:
                    #     row_analysis = row_analysis.split(', ')[0].strip()
                    # except:
                    #     row_analysis = row_analysis
                    self.jailbreak_db_analysis.append(row_analysis)
                if 'red_flags' in row:
                    self.jailbreak_db_red_flags.append(row['red_flags'])
                if 'guidance' in row:
                    row_guidance = row['guidance']
                    try:
                        row_guidance = row_guidance.split('Guidance: ')[1].strip()
                    except:
                        row_guidance = row_guidance
                    self.jailbreak_db_guidance.append(row_guidance)
        # elif jailbreak_db_index_file is None and run_index[0] is False:
            # raise ValueError('specify existing jailbreak database index file or run_index[0] must be True')
        elif run_index[0] is True:
            assert jailbreak_db_source_file is not None
            print(f'Building jailbreak database from {jailbreak_db_source_file}...')
            jailbreak_db_source_df = pd.read_csv(jailbreak_db_source_file)
            self.jailbreak_db_source = []
            self.jailbreak_db_seed_queries = []
            self.jailbreak_db_scheme = []
            self.jailbreak_db_analysis = []
            self.jailbreak_db_red_flags = []
            self.jailbreak_db_guidance = []
            for _, row in jailbreak_db_source_df.iterrows():
                self.jailbreak_db_source.append(row['jailbreak'])
                self.jailbreak_db_seed_queries.append(row['query'])
                if 'scheme' in row:
                    self.jailbreak_db_scheme.append(row['scheme'])
                if 'analysis' in row:
                    row_analysis = row['analysis']
                    try:
                        row_analysis = row_analysis.split('Analysis: ')[1].strip()
                    except:
                        row_analysis = row_analysis
                    try:
                        row_analysis = row_analysis.split(', ')[0].strip()
                    except:
                        row_analysis = row_analysis
                    self.jailbreak_db_analysis.append(row_analysis)
                if 'red_flags' in row:
                    self.jailbreak_db_red_flags.append(row['red_flags'])
                if 'guidance' in row:
                    row_guidance = row['guidance']
                    try:
                        row_guidance = row_guidance.split('Guidance: ')[1].strip()
                    except:
                        row_guidance = row_guidance
                    self.jailbreak_db_guidance.append(row_guidance)
            self._initialize_faiss('jailbreak')
            jailbreak_db_output_file = jailbreak_db_index_file or jailbreak_db_source_file.split(".")[0] + '.index'
            print(f'Saving jailbreak database to {jailbreak_db_output_file}...')
            faiss.write_index(self.jailbreak_db, jailbreak_db_output_file)

        if query_db_index_file is not None and run_index[1] is False:
            assert query_db_source_file is not None
            print(f'Loading query database from {query_db_index_file}...')
            self.query_db = faiss.read_index(query_db_index_file)
            with open(query_db_source_file, 'r') as f:
                self.query_db_source = [line.strip() for line in f.readlines()]
        # elif query_db_index_file is None and run_index[1] is False:
            # raise ValueError('specify existing query database index file or run_index[1] must be True')
        elif run_index[1] is True:
            assert query_db_source_file is not None
            print(f'Building query database from {query_db_source_file}...')
            with open(query_db_source_file, 'r') as f:
                self.query_db_source = [line.strip() for line in f.readlines()]
            self._initialize_faiss('query')
            query_db_output_file = query_db_index_file or query_db_source_file.split(".")[0] + '.index'
            print(f'Saving query database to {query_db_output_file}...')
            faiss.write_index(self.query_db, query_db_output_file)
        
        self.no_strategy_examples = [
            'The user prompt is straightforward and genuine in expressing the query.',
            'The user prompt is straightforward and ethical in expressing the intent.',
            'The user prompt does not use any strategy to obscure the intent as it directly asks for the same information without any indirect phrasing or additional context.',
            'The prompt does not use any strategy to obscure the intent as it directly asks the same question without any linguistic cues, structural choices, or context framing to hide its true purpose.',
            'The prompt uses direct and common phrasing without additional context or complexity, making it difficult to identify any obfuscation strategy since the intent is clearly stated and not obscured.',
            'The following prompt is straightforward in expressing the intent.',
            'The following prompt clearly conveys its intent.',
            'The following prompt directly conveys the intent.',
            'The following prompt is unambiguous in its presentation of the intent.',
            'This prompt presents the intent in a straightforward manner.',
            'The prompt does not employ any obfuscation strategy, as it states the intent directly and without contextual framing or linguistic disguise.'
        ]

    def _initialize_faiss(self, index_name):
        if index_name == 'jailbreak':
            sentences = self.jailbreak_db_source
            embeddings = self.retrievers[0].get_embeddings(sentences)
        elif index_name == 'query':
            sentences = self.query_db_source
            embeddings = self.retrievers[1].get_embeddings(sentences)
        dimension = embeddings.shape[1]
        faiss.normalize_L2(embeddings)
        if index_name == 'jailbreak':
            self.jailbreak_db = faiss.IndexFlatIP(dimension)
            self.jailbreak_db.add(embeddings)
        elif index_name == 'query':
            self.query_db = faiss.IndexFlatIP(dimension)
            self.query_db.add(embeddings)

    def _query_index(self, query, index, k=5):
        if index == 'jailbreak':
            query_embeddings = self.retrievers[0].get_embeddings([query])
            faiss.normalize_L2(query_embeddings)
            distances, indices = self.jailbreak_db.search(query_embeddings, k)
            return distances, indices
        elif index == 'query':
            query_embeddings = self.retrievers[1].get_embeddings([query])
            faiss.normalize_L2(query_embeddings)
            distances, indices = self.query_db.search(query_embeddings, k)
            return distances, indices

    def generate(self, prompt, precompute_index=-1):
        if precompute_index >= 0 and self.precompute_df is not None:
            precompute_row = self.precompute_df.iloc[precompute_index]
            pmi_hypotheses = self._safe_list_eval(precompute_row['pmi_hypotheses'])
            intent_posterior_hypotheses = self._safe_list_eval(precompute_row['intent_posterior_hypotheses'])
            softmax_hypotheses = self._safe_list_eval(precompute_row['softmax_hypotheses'])
            classifier_probs = self._safe_list_eval(precompute_row['classifier_probs'])
            label = precompute_row['label']
            if len(pmi_hypotheses) > 0:
                pmi_tensor = torch.tensor(pmi_hypotheses, dtype=torch.float32).to(self.device)
                intent_posterior_tensor = torch.tensor(intent_posterior_hypotheses, dtype=torch.float32).to(self.device)
                softmax_tensor = torch.tensor(softmax_hypotheses, dtype=torch.float32).to(self.device)
                cls_probs_tensor = torch.tensor(classifier_probs, dtype=torch.float32).to(self.device)
                if not self.uni_posterior:
                    cls_probs_tensor = cls_probs_tensor * softmax_tensor
                    cls_prob = cls_probs_tensor.sum().item()  # Move back to CPU for final result
                else:
                    cls_prob = cls_probs_tensor.sum().item() / cls_probs_tensor.shape[0]  # Move back to CPU for final result
            else:
                cls_prob = classifier_probs[0]
            if cls_prob > self.cls_threshold:
                # print(f'****Classifier decision: 1')
                response = 'I\'m sorry, but I can\'t help with that.'
                eval_metrics1 = {
                    'refusal': 1,
                    'convincingness': 0,
                    'specificity': 0,
                    'score': 0,
                }
                eval_metrics2 = {
                    'eval': 'refusal'
                }
                if self.precompute_response_df is not None:
                    precompute_response_row = self.precompute_response_df.iloc[precompute_index]
                    if 'refusal' in precompute_response_row:
                        return response, eval_metrics1
                    else:
                        return response, eval_metrics2
            else:
                # print(f'****Classifier decision: 0')
                if self.precompute_response_df is not None:
                    precompute_response_row = self.precompute_response_df.iloc[precompute_index]
                    if 'refusal' in precompute_response_row:
                        response = precompute_response_row['response']
                        refusal = precompute_response_row['refusal']
                        convincingness = precompute_response_row['convincingness']
                        specificity = precompute_response_row['specificity']
                        score = precompute_response_row['score']
                        eval_metrics = {
                            'refusal': refusal,
                            'convincingness': convincingness,
                            'specificity': specificity,
                            'score': score,
                        }
                    else:
                        response = precompute_response_row['response']
                        eval_dict = ast.literal_eval(precompute_response_row['eval'])
                        eval_metrics = {
                            'eval': eval_dict['eval']
                        }
                    return response, eval_metrics
                else:
                    sys_prompt = "You are a helpful assistant"
                    response = self.llm.generate_response(prompt, sys_prompt=sys_prompt)
            return response
        if self.test_test:
            return self.generate_clean(prompt)
        if self.clean:
            return self.generate_clean(prompt)
        if self.clean_norm:
            return self.generate_clean_norm(prompt)
        self.counter += 1
        retriever_scores, jb_indices = self._query_index(prompt, 'jailbreak', 50)
        retriever_scores = retriever_scores[0]
        jb_indices = jb_indices[0]
        reranker_scores = []
        reranker_contexts = []
        best_no_strategy_example = None
        if self.paraphrase_no_strategy:
            if self.strategy_rerank:
                no_strategy_reranker_context = f'You are provided with an user prompt.\n\nUser Prompt: {prompt}\n\nThe following analysis accurately describes the patterns and framings presented in the user prompt.\n\nAnalysis: '
                no_strategy_reranker_scores = []
                for no_strategy_example in self.no_strategy_examples:
                    no_strategy_reranker_scores.append(self.reranker.rerank(no_strategy_example, no_strategy_reranker_context, relative_llh=False, length_norm=True))
                no_strategy_examples_scores = [(no_strategy_reranker_score, no_strategy_example) for no_strategy_reranker_score, no_strategy_example in zip(no_strategy_reranker_scores, self.no_strategy_examples)]
                no_strategy_examples_scores.sort(key=lambda x: x[0], reverse=True)
                best_no_strategy_example = no_strategy_examples_scores[0][1]
            else:
                no_strategy_reranker_scores = []
                for no_strategy_example in self.no_strategy_examples:
                    no_strategy_reranker_context = f'The following strategy accurately describes the patterns of the user prompt:\nStrategy: {no_strategy_example}\nUser prompt: '
                    no_strategy_reranker_scores.append(self.reranker.rerank(prompt, no_strategy_reranker_context, relative_llh=False, length_norm=True))
                no_strategy_examples_scores = [(no_strategy_reranker_score, no_strategy_example) for no_strategy_reranker_score, no_strategy_example in zip(no_strategy_reranker_scores, self.no_strategy_examples)]
                no_strategy_examples_scores.sort(key=lambda x: x[0], reverse=True)
                best_no_strategy_example = no_strategy_examples_scores[0][1]
        for jb_index in jb_indices:
            jb_example = self.jailbreak_db_source[jb_index]
            intent_example = self.jailbreak_db_seed_queries[jb_index]
            strategy_example = self.jailbreak_db_analysis[jb_index]
            if strategy_example[0] == '"':
                strategy_example = strategy_example[1:]
            if strategy_example[-1] == '"':
                strategy_example = strategy_example[:-1]
            strategy_example = strategy_example.replace("The prompt", "The user prompt")
            strategy_example = strategy_example.replace("the prompt", 'the user prompt')
            strategy_example = strategy_example.replace("This prompt", "The user prompt")
            strategy_example = strategy_example.replace("this prompt", "the user prompt")
            if strategy_example.startswith('Analysis: '):
                strategy_example = strategy_example[10:]
            if self.strategy_rerank:
                reranker_context = f'You are provided with an user prompt.\n\nUser Prompt: {prompt}\n\nThe following analysis accurately describes the patterns and framings presented in the user prompt.\n\nAnalysis: '
                if self.new_prompt:
                    reranker_context = f'The patterns and wordings of the following user prompt is accurately described and analyzed by the following analysis.\n\nUser prompt: {prompt}\nAnalysis: '
                reranker_contexts.append(reranker_context)
                reranker_scores.append(self.reranker.rerank(strategy_example, reranker_context, relative_llh=False, length_norm=True))
            else:
                reranker_context = f'The following strategy accurately describes the patterns of the user prompt:\nStrategy: {strategy_example}\nUser prompt: '
                # reranker_context = f'You are given an analysis of an user prompt.\n\nAnalysis:\n{strategy_example}\n\nThe pattern and framings of the following user prompt aligns with the description in the analysis.\n\nUser prompt: '
                reranker_contexts.append(reranker_context)
                reranker_scores.append(self.reranker.rerank(prompt, reranker_context, relative_llh=False, length_norm=True))
        # reranker_scores = self.reranker.rerank(prompt, reranker_contexts, relative_llh=False, length_norm=True)
        if self.strategy_rerank or self.no_rerank:
            if best_no_strategy_example is not None:
                no_strategy_example = best_no_strategy_example
            else:
                no_strategy_example = 'The user prompt is straightforward and genuine in expressing the query.'
            base_reranker_context = reranker_context
            base_reranker_score = self.reranker.rerank(no_strategy_example, reranker_context, relative_llh=False, length_norm=True)
            # pmi_list = [base_reranker_score for rrk_score in reranker_scores]
            pmi_list = reranker_scores
            if self.no_rerank:
                pmi_list = [base_reranker_score for rrk_score in reranker_scores]
            relative_pmi_list = [rrk_score - base_reranker_score for rrk_score in reranker_scores]
            rerank_items = [(jb_index, pmi) for jb_index, pmi in zip(jb_indices, pmi_list)]
            pmi_threshold = 0.0
            rerank_items = [rerank_item for rerank_item, relative_pmi in zip(rerank_items, relative_pmi_list) if relative_pmi > pmi_threshold]
        else:
            base_reranker_context = 'User prompt: '
            base_reranker_score = self.reranker.rerank(prompt, base_reranker_context, relative_llh=False, length_norm=True)
            pmi_list = [rrk_score - base_reranker_score for rrk_score in reranker_scores]
            rerank_items = [(jb_index, pmi) for jb_index, pmi in zip(jb_indices, pmi_list)]
            pmi_threshold = 0.0
            rerank_items = [rerank_item for rerank_item in rerank_items if rerank_item[1] > pmi_threshold]
        if len(rerank_items) > 0:
            rerank_items.sort(key=lambda x: x[1], reverse=True)
            print(f'****Count: {len(rerank_items)}')
            rerank_k = min(10, len(rerank_items))
            rerank_items = rerank_items[:rerank_k]
            extractor_examples = []
            for jb_index, pmi in rerank_items:
                strategy_example = self.jailbreak_db_analysis[jb_index]
                if strategy_example[0] == '"':
                    strategy_example = strategy_example[1:]
                if strategy_example[-1] == '"':
                    strategy_example = strategy_example[:-1]
                strategy_example = strategy_example.replace("The prompt", "The user prompt")
                strategy_example = strategy_example.replace("the prompt", 'the user prompt')
                strategy_example = strategy_example.replace("This prompt", "The user prompt")
                strategy_example = strategy_example.replace("this prompt", "the user prompt")
                if strategy_example.startswith('Analysis: '):
                    strategy_example = strategy_example[10:]
                guidance_example = self.jailbreak_db_guidance[jb_index]
                extractor_examples.append((strategy_example, guidance_example))
            if best_no_strategy_example is not None:
                no_strategy_example = best_no_strategy_example
            else:
                no_strategy_example = 'The user prompt is straightforward and genuine in expressing the query.'
            # no_strategy_example = 'The prompt does not use any strategy to obscure the intent as it directly asks for the same information without any indirect phrasing or additional context.'
            # no_strategy_example = 'The user prompt is straightforward and ethical in expressing the intent.'
            no_strategy_guidance = "The intent can be taken by directly summarizing the prompt."
            # no_strategy_guidance = "The intent can be taken by directly repeating the prompt."
            no_strategy_reranker_context = f'The following strategy accurately describes the patterns of the user prompt:\nStrategy: {no_strategy_example}\nUser prompt: '
            # no_strategy_reranker_context = f'You are given an analysis of an user prompt.\n\nAnalysis:\n{no_strategy_example}\n\nThe pattern and framings of the following user prompt aligns with the description in the analysis.\n\nUser prompt: '
            no_strategy_reranker_score = self.reranker.rerank(prompt, no_strategy_reranker_context, relative_llh=False, length_norm=True)
            if self.strategy_rerank or self.no_rerank:
                no_strategy_pmi = base_reranker_score
            else:
                no_strategy_pmi = no_strategy_reranker_score - base_reranker_score
            rerank_items.append((-1, no_strategy_pmi))
            print(f'****Rerank items: {rerank_items}')
            extractor_examples.append((no_strategy_example, no_strategy_guidance))
            intents_raw = self.extractor.extract(prompt, extractor_examples, prompt_v18=False, return_selection=False)
            intents = []
            for intent_idx, intent_raw in enumerate(intents_raw):
                if intent_raw.startswith(': \n'):
                    intent_raw = intent_raw[3:].strip()
                if intent_raw.startswith(': '):
                    intent_raw = intent_raw[2:].strip()
                intents.append(intent_raw)
                print(f'****Extracted intent {intent_idx}: {intent_raw}')
            no_strategy_intent = intents[-1]
            if len(intents) > 0:
                intent_posterior_list = []
                intent_posterior_contexts = []
                intent_hypotheses = []
                strategy_posterior_list = []
                for intent_hypothesis, strategy_guidance_hypothesis in zip(intents, extractor_examples):
                    strategy_hypothesis, guidance_hypothesis = strategy_guidance_hypothesis
                    # intent_posterior_context = f'The user applies the following strategy to construct the user prompt that disguises the underlying intent.\n\nStrategy: {strategy_hypothesis}\n\nUser prompt: {prompt}\n\nThe underlying intent is: '
                    # intent_posterior_context = f'The user applies the following strategy to construct the user prompt that disguises the underlying intent.\n\nStrategy: {strategy_hypothesis}\n\nUser prompt: {prompt}\n\nThe underlying intent is extracted from the user prompt according to the strategy.\n\nIntent: '
                    intent_posterior_context = f'You are an Intent Extraction Assistant. Your primary task is to clearly identify the user\'s actual intent based on the provided user prompt.\n\nUser Prompt:\n{prompt}\n\nYou are given a strategy that explains how the user prompt could be constructed from the user\'s actual intent.\n\nStrategy:\n{strategy_hypothesis}\n\nYour Tasks:\n1. Analyze how the strategy explains the construction of the user prompt.\n2. Extract the user\'s actual intent according to the strategy.\n\nOutput:\nExtracted Intent: '
                    intent_posterior_contexts.append(intent_posterior_context)
                    intent_hypotheses.append(intent_hypothesis)
                    intent_posterior = self.reranker.rerank(intent_hypothesis, intent_posterior_context, relative_llh=False, length_norm=True)
                    intent_posterior_list.append(intent_posterior)
                    strategy_posterior_context = f'You are provided with an user prompt.\n\nUser Prompt: {prompt}\n\nThe following analysis accurately describes the patterns and framings presented in the user prompt.\n\nAnalysis: '
                    if self.new_prompt:
                        strategy_posterior_context = f'The patterns and wordings of the following user prompt is accurately described and analyzed by the following analysis.\n\nUser prompt: {prompt}\nAnalysis: '
                    strategy_posterior = self.reranker.rerank(strategy_hypothesis, strategy_posterior_context, relative_llh=False, length_norm=True)
                    strategy_posterior_list.append(strategy_posterior)
                # intent_posterior_list = self.reranker.rerank(intent_hypotheses, intent_posterior_contexts, relative_llh=False, length_norm=True)
                print(f'****Intent posterior list: {intent_posterior_list}')
                # intent_posterior_tensor = torch.tensor(intent_posterior_list, dtype=torch.float32)
                # intent_posterior_tensor = torch.nn.functional.log_softmax(intent_posterior_tensor, dim=0)
                # pmi_tensor = torch.tensor([rerank_item[1] for rerank_item in rerank_items], dtype=torch.float32)
                # pmi_tensor = torch.nn.functional.log_softmax(pmi_tensor, dim=0)
                # print(f'****PMI tensor: {pmi_tensor}')
                # print(f'****Intent posterior tensor: {intent_posterior_tensor}')
                # intent_strategy_posterior_tensor = intent_posterior_tensor + pmi_tensor
                # softmax_probs = torch.nn.functional.softmax(intent_strategy_posterior_tensor, dim=0)


                intent_posterior_tensor = torch.tensor(intent_posterior_list, dtype=torch.float32)
                intent_posterior_tensor = torch.nn.functional.log_softmax(intent_posterior_tensor, dim=0)
                if self.no_pmi:
                    pmi_tensor = torch.tensor(strategy_posterior_list, dtype=torch.float32)
                else:
                    pmi_tensor = torch.tensor([rerank_item[1] for rerank_item in rerank_items], dtype=torch.float32)
                pmi_tensor = torch.nn.functional.log_softmax(pmi_tensor, dim=0)
                print(f'****PMI tensor: {pmi_tensor}')
                print(f'****Intent posterior tensor: {intent_posterior_tensor}')
                intent_strategy_posterior_tensor = intent_posterior_tensor + pmi_tensor
                # exponantial of intent_strategy_posterior_tensor
                # softmax_probs = torch.exp(intent_strategy_posterior_tensor)
                softmax_probs = torch.nn.functional.softmax(intent_strategy_posterior_tensor, dim=0)

                print(f'****Softmax probabilities: {softmax_probs}')
                cls_probs = []
                no_probs = []
                intent_hypotheses = []
                strategy_hypotheses = []
                for intent_hypothesis, strategy_guidance_hypothesis in zip(intents, extractor_examples):
                    strategy_hypothesis, guidance_hypothesis = strategy_guidance_hypothesis
                    intent_hypotheses.append(intent_hypothesis)
                    strategy_hypotheses.append(strategy_hypothesis)
                    # strategy_hypotheses.append('')
                    # yes_prob, no_prob = self.classifier.classify(prompt, intent_hypothesis, strategy_hypothesis, output_logits=True)
                    # cls_probs.append(yes_prob)
                    # no_probs.append(no_prob)
                if self.no_strategy_classify:
                    cls_probs, no_probs = self.classifier.classify(prompt, intent_hypotheses, strategy=None, output_logits=True)
                elif self.intent_only_classify:
                    cls_probs, no_probs = self.classifier.classify(prompt, intent_hypotheses, strategy=None, output_logits=True, intent_only=True)
                else:
                    cls_probs, no_probs = self.classifier.classify(prompt, intent_hypotheses, strategy_hypotheses, output_logits=True)
                print(f'****Classifier probabilities: {cls_probs}')
                cls_probs_tensor = torch.tensor(cls_probs, dtype=torch.float32)
                no_probs_tensor = torch.tensor(no_probs, dtype=torch.float32)
                if not self.uni_posterior:
                    cls_probs_tensor = cls_probs_tensor * softmax_probs
                    no_probs_tensor = no_probs_tensor * softmax_probs
                    # cls_probs_tensor: [cls_prob1, cls_prob2, cls_prob3, cls_prob4, cls_prob5]
                    # no_probs_tensor: [no_prob1, no_prob2, no_prob3, no_prob4, no_prob5]
                    # pair_tensor = torch.stack([cls_probs_tensor, no_probs_tensor], dim=1)
                    # pair_tensor = torch.nn.functional.softmax(pair_tensor, dim=1)
                    # cls_probs_tensor = pair_tensor[:, 0]
                    # no_probs_tensor = pair_tensor[:, 1]
                    cls_prob = cls_probs_tensor.sum()#  / cls_probs_tensor.shape[0]
                    no_prob = no_probs_tensor.sum()#  / no_probs_tensor.shape[0]
                    summed_prob = cls_prob + no_prob
                else:
                    cls_prob = cls_probs_tensor.sum() / cls_probs_tensor.shape[0]
                    no_prob = no_probs_tensor.sum() / no_probs_tensor.shape[0]
                    summed_prob = cls_prob + no_prob
                print(f'****Classifier final probability: {cls_prob}')
                print(f'****Classifier no probability: {no_prob}')
                print(f'****Classifier summed probability: {summed_prob}')
                if cls_prob > self.cls_threshold:
                    print(f'****Classifier decision: 1')
                    response = 'I\'m sorry, but I can\'t help with that.'
                    if self.precompute_response_df is not None:
                        precompute_response_row = self.precompute_response_df.iloc[precompute_index]
                        if 'refusal' in precompute_response_row:
                            eval_metrics = {
                                'refusal': 1,
                                'convincingness': 0,
                                'specificity': 0,
                                'score': 0,
                            }
                        else:
                            eval_metrics = {
                                'eval': 'refusal'
                            }
                else:
                    print(f'****Classifier decision: 0')
                    if self.precompute_response_df is not None:
                        precompute_response_row = self.precompute_response_df.iloc[precompute_index]
                        if 'refusal' in precompute_response_row:
                            response = precompute_response_row['response']
                            refusal = precompute_response_row['refusal']
                            convincingness = precompute_response_row['convincingness']
                            specificity = precompute_response_row['specificity']
                            score = precompute_response_row['score']
                            eval_metrics = {
                                'refusal': refusal,
                                'convincingness': convincingness,
                                'specificity': specificity,
                                'score': score,
                            }
                        else:
                            response = precompute_response_row['response']
                            eval_dict = ast.literal_eval(precompute_response_row['eval'])
                            eval_metrics = {
                                'eval': eval_dict['eval']
                            }
                    else:
                        sys_prompt = "You are a helpful assistant"
                        response = self.llm.generate_response(prompt, sys_prompt=sys_prompt)
                print('='*100)
                scores_to_log = {
                    'request_id': str(self.counter),
                    'prompt': prompt,
                    'pmi_hypotheses': str(pmi_tensor.tolist()),
                    'intent_posterior_hypotheses': str(intent_posterior_tensor.tolist()),
                    'softmax_hypotheses': str(softmax_probs.tolist()),
                    'classifier_probs': str(cls_probs),
                    'label': self.output_label,
                }

                if self.scores_output_path is not None:
                    # File already exists and header written in __init__
                    # Open in append mode
                    with open(self.scores_output_path, 'a', newline='') as f_out:
                        writer = csv.DictWriter(f_out, fieldnames=['request_id', 'prompt', 'pmi_hypotheses', 'intent_posterior_hypotheses', 'softmax_hypotheses', 'classifier_probs', 'label'])
                        writer.writerow(scores_to_log)
                if self.precompute_response_df is not None:
                    return response, eval_metrics
                else:
                    return response
            else:
                # TODO: add classifier score
                if self.strategy_classify:
                    yes_prob, no_prob = self.classifier.classify(prompt, no_strategy_intent, no_strategy_example, output_logits=True)
                else:
                    yes_prob, no_prob = self.classifier.classify(prompt, no_strategy_intent, strategy=None, output_logits=True)
                cls_probs = [yes_prob]
                print(f'****Classifier probabilities: {cls_probs}')
                
                if yes_prob > self.cls_threshold:
                    response = 'I\'m sorry, but I can\'t help with that.'
                    print(f'****Classifier decision (no RA): 1')
                else:
                    sys_prompt = "You are a helpful assistant"
                    response = self.llm.generate_response(prompt, sys_prompt=sys_prompt)
                    print(f'****Classifier decision (no RA): 0')
                print('='*100)
                return response
        else:
            if self.paraphrase_no_strategy:
                no_strategy_example = best_no_strategy_example
            else:
                no_strategy_example = 'The user prompt is straightforward and genuine in expressing the query.'
            # no_strategy_example = 'The prompt does not use any strategy to obscure the intent as it directly asks for the same information without any indirect phrasing or additional context.'
            no_strategy_guidance = "The intent can be taken by directly summarizing the prompt."
            no_strategy_intent = self.extractor.extract(prompt, [(no_strategy_example, no_strategy_guidance)], prompt_v18=False, return_selection=False)[0]
            if no_strategy_intent.startswith(': \n'):
                no_strategy_intent = no_strategy_intent[3:].strip()
            if no_strategy_intent.startswith(': '):
                no_strategy_intent = no_strategy_intent[2:].strip()
            if self.no_strategy_classify:
                yes_prob, no_prob = self.classifier.classify(prompt, no_strategy_intent, strategy=None, output_logits=True)
            elif self.intent_only_classify:
                yes_prob, no_prob = self.classifier.classify(prompt, no_strategy_intent, strategy=None, output_logits=True, intent_only=True)
            else:
                yes_prob, no_prob = self.classifier.classify(prompt, no_strategy_intent, no_strategy_example, output_logits=True)
            cls_probs = [yes_prob]
            print(f'****Classifier probabilities: {cls_probs}')
            if yes_prob > self.cls_threshold:
                response = 'I\'m sorry, but I can\'t help with that.'
                print(f'****Classifier decision (no RA): 1')
                if self.precompute_response_df is not None:
                    precompute_response_row = self.precompute_response_df.iloc[precompute_index]
                    if 'refusal' in precompute_response_row:
                        eval_metrics = {
                            'refusal': 1,
                            'convincingness': 0,
                            'specificity': 0,
                            'score': 0,
                        }
                    else:
                        eval_metrics = {
                            'eval': 'refusal'
                        }
            else:
                if self.precompute_response_df is not None:
                    precompute_response_row = self.precompute_response_df.iloc[precompute_index]
                    response = precompute_response_row['response']
                    if 'refusal' in precompute_response_row:
                        refusal = precompute_response_row['refusal']
                        convincingness = precompute_response_row['convincingness']
                        specificity = precompute_response_row['specificity']
                        score = precompute_response_row['score']
                        eval_metrics = {
                            'refusal': refusal,
                            'convincingness': convincingness,
                            'specificity': specificity,
                            'score': score,
                        }
                    else:
                        eval_dict = ast.literal_eval(precompute_response_row['eval'])
                        eval_metrics = {
                            'eval': eval_dict['eval']
                        }
                else:
                    sys_prompt = "You are a helpful assistant"
                    response = self.llm.generate_response(prompt, sys_prompt=sys_prompt)
                print(f'****Classifier decision (no RA): 0')

            print('='*100)
            
            if self.scores_output_path is not None:
                scores_to_log = {
                    'request_id': str(self.counter),
                    'prompt': prompt,
                    'pmi_hypotheses': str([]),  # Empty list for scores
                    'intent_posterior_hypotheses': str([]),
                    'softmax_hypotheses': str([]),
                    'classifier_probs': str(cls_probs), # No classification score calculated here
                    'label': self.output_label,
                }
                with open(self.scores_output_path, 'a', newline='') as f_out:
                    writer = csv.DictWriter(f_out, fieldnames=['request_id', 'prompt', 'pmi_hypotheses', 'intent_posterior_hypotheses', 'softmax_hypotheses', 'classifier_probs', 'label'])
                    writer.writerow(scores_to_log)
            if self.precompute_response_df is not None:
                return response, eval_metrics
            else:
                return response

    def generate_non_bayesian(self, prompt):
        retriever_scores, jb_indices = self._query_index(prompt, 'jailbreak', 50)
        retriever_scores = retriever_scores[0]
        jb_indices = jb_indices[0]
        reranker_scores = []
        for jb_index in jb_indices:
            jb_example = self.jailbreak_db_source[jb_index]
            intent_example = self.jailbreak_db_seed_queries[jb_index]
            strategy_example = self.jailbreak_db_analysis[jb_index]
            if strategy_example[0] == '"':
                strategy_example = strategy_example[1:]
            if strategy_example[-1] == '"':
                strategy_example = strategy_example[:-1]
            strategy_example = strategy_example.replace("The prompt", "The user prompt")
            strategy_example = strategy_example.replace("the prompt", 'the user prompt')
            strategy_example = strategy_example.replace("This prompt", "The user prompt")
            strategy_example = strategy_example.replace("this prompt", "the user prompt")
            if strategy_example.startswith('Analysis: '):
                strategy_example = strategy_example[10:]
            reranker_context = f'The following strategy describes the user prompt:\nStrategy: {strategy_example}\nUser prompt: '
            reranker_scores.append(self.reranker.rerank(prompt, reranker_context, relative_llh=False, length_norm=True))
        base_reranker_context = 'User prompt: '
        base_reranker_score = self.reranker.rerank(prompt, base_reranker_context, relative_llh=False, length_norm=True)
        pmi_list = [rrk_score - base_reranker_score for rrk_score in reranker_scores]
        rerank_items = [(jb_index, pmi) for jb_index, pmi in zip(jb_indices, pmi_list)]
        pmi_threshold = 0.0
        rerank_items = [rerank_item for rerank_item in rerank_items if rerank_item[1] > pmi_threshold]
        if len(rerank_items) > 0:
            rerank_items.sort(key=lambda x: x[1], reverse=True)
            print(f'****Count: {len(rerank_items)}')
            rerank_k = min(5, len(rerank_items))
            rerank_items = rerank_items[:rerank_k]
            extractor_examples = []
            for jb_index, pmi in rerank_items:
                strategy_example = self.jailbreak_db_analysis[jb_index]
                guidance_example = self.jailbreak_db_guidance[jb_index]
                extractor_examples.append((strategy_example, guidance_example))
                no_strategy_example = 'The prompt is straightforward in expressing the intent without any strategies.'
                no_strategy_guidance = "The intent can be taken by directly summarizing the prompt."
                extractor_examples.append((no_strategy_example, no_strategy_guidance))
            intents_raw, selection = self.extractor.extract(prompt, extractor_examples, prompt_v18=True, return_selection=True)
            intents = []
            for intent_idx, intent_raw in enumerate(intents_raw):
                if intent_raw.startswith(': \n'):
                    intent_raw = intent_raw[3:].strip()
                if intent_raw.startswith(': '):
                    intent_raw = intent_raw[2:].strip()
                intents.append(intent_raw)
                print(f'****Extracted intent {intent_idx}: {intent_raw}')
            print(f'****Selection: {selection}')
            no_strategy_intent = intents[-1]
            extractor_selected_intents = [intent for intent, sel in zip(intents, selection) if sel == 1]
            if len(extractor_selected_intents) > 0:
                intent = extractor_selected_intents[0]
            else:
                intent = no_strategy_intent
        else:
            extractor_examples = [('', '')]
            intent = self.extractor.extract(prompt, extractor_examples)[0]
        print(f'****Final intent: {intent}')
        decision = self.classifier.classify(intent)
        print(f'****Classifier decision: {decision}')
        if decision == 1:
            response = 'I\'m sorry, but I can\'t help with that.'
        else:
            sys_prompt = "You are a helpful assistant"
            response = self.llm.generate_response(prompt, sys_prompt=sys_prompt)
        return response, intent

    def generate_two_rerank(self, prompt):
        if self.test_test:
            return self.generate_clean(prompt)
        if self.clean:
            return self.generate_clean(prompt)
        if self.clean_norm:
            return self.generate_clean_norm(prompt)
        retriever_scores, jb_indices = self._query_index(prompt, 'jailbreak', 50)
        retriever_scores = retriever_scores[0]
        jb_indices = jb_indices[0]
        reranker_scores = []
        for jb_index in jb_indices:
            jb_example = self.jailbreak_db_source[jb_index]
            intent_example = self.jailbreak_db_seed_queries[jb_index]
            strategy_example = self.jailbreak_db_analysis[jb_index]
            if strategy_example[0] == '"':
                strategy_example = strategy_example[1:]
            if strategy_example[-1] == '"':
                strategy_example = strategy_example[:-1]
            strategy_example = strategy_example.replace("The prompt", "The user prompt")
            strategy_example = strategy_example.replace("the prompt", 'the user prompt')
            strategy_example = strategy_example.replace("This prompt", "The user prompt")
            strategy_example = strategy_example.replace("this prompt", "the user prompt")
            if strategy_example.startswith('Analysis: '):
                strategy_example = strategy_example[10:]
            incomplete_user_prompt = prompt[:-1]
            reranker_context = f'The following strategy accurately describes the patterns of the user prompt:\nStrategy: {strategy_example}\nUser prompt: '
            # reranker_context = f'The following strategy accurately describes the patterns of the user prompt:\nStrategy: {strategy_example}\nEvaluate the user prompt: {prompt}\n'
            reranker_scores.append(self.reranker.rerank(prompt, reranker_context, relative_llh=False, length_norm=True))
            # reranker_scores.append(self.reranker.rerank("Evaluation", reranker_context, relative_llh=False, length_norm=True))
        base_reranker_context = 'User prompt: '
        base_reranker_score = self.reranker.rerank(prompt, base_reranker_context, relative_llh=False, length_norm=True)
        # base_reranker_context = f'Evaluate the user prompt: {prompt}\n'
        # print(f'****Prompt Last Token: {prompt[-1]}')
        # base_reranker_score = self.reranker.rerank("Evaluation", base_reranker_context, relative_llh=False, length_norm=True)
        pmi_list = [rrk_score - base_reranker_score for rrk_score in reranker_scores]
        rerank_items = [(jb_index, pmi) for jb_index, pmi in zip(jb_indices, pmi_list)]
        pmi_threshold = 0.0
        rerank_items = [rerank_item for rerank_item in rerank_items if rerank_item[1] > pmi_threshold]
        if len(rerank_items) > 0:
            rerank_items.sort(key=lambda x: x[1], reverse=True)
            print(f'****Count: {len(rerank_items)}')
            rerank_k = min(10, len(rerank_items))
            rerank_items = rerank_items[:rerank_k]
            extractor_examples = []
            for jb_index, pmi in rerank_items:
                strategy_example = self.jailbreak_db_analysis[jb_index]
                guidance_example = self.jailbreak_db_guidance[jb_index]
                extractor_examples.append((strategy_example, guidance_example))
            # no_strategy_example = 'The prompt is straightforward in expressing the intent without any strategies.'
            no_strategy_example = 'The prompt is straightforward and genuine in expressing the query.'
            no_strategy_guidance = "The intent can be taken by directly summarizing the prompt."
            no_strategy_reranker_context = f'The following strategy accurately describes the user prompt:\nStrategy: {no_strategy_example}\nUser prompt: '
            no_strategy_reranker_score = self.reranker.rerank(prompt, no_strategy_reranker_context, relative_llh=False, length_norm=True)
            # incomplete_user_prompt = prompt[:-1]
            # no_strategy_reranker_context = f'The following strategy accurately describes the user prompt:\nStrategy: {no_strategy_example}\nEvaluate the user prompt: {prompt}\n'
            # no_strategy_reranker_score = self.reranker.rerank("Evaluation", no_strategy_reranker_context, relative_llh=False, length_norm=True)
            no_strategy_pmi = no_strategy_reranker_score - base_reranker_score
            rerank_items.append((-1, no_strategy_pmi))
            print(f'****Rerank items: {rerank_items}')
            extractor_examples.append((no_strategy_example, no_strategy_guidance))
            intents_raw, selection = self.extractor.extract(prompt, extractor_examples, prompt_v18=True, return_selection=True)
            intents = []
            for intent_idx, intent_raw in enumerate(intents_raw):
                if intent_raw.startswith(': \n'):
                    intent_raw = intent_raw[3:].strip()
                if intent_raw.startswith(': '):
                    intent_raw = intent_raw[2:].strip()
                intents.append(intent_raw)
                print(f'****Extracted intent {intent_idx}: {intent_raw}')
            print(f'****Selection: {selection}')
            no_strategy_intent = intents[-1]
            intents = [intent for intent, sel in zip(intents[:-1], selection[:-1]) if sel == 1]
            if len(intents) > 0:
                intents.append(no_strategy_intent)
                extractor_examples = [extractor_example for extractor_example, sel in zip(extractor_examples[:-1], selection[:-1]) if sel == 1]
                extractor_examples.append((no_strategy_example, no_strategy_guidance))
                rerank_items = [rerank_item for rerank_item, sel in zip(rerank_items[:-1], selection[:-1]) if sel == 1]
                rerank_items.append((-1, no_strategy_pmi))
                intent_posterior_list = []
                for intent_hypothesis, strategy_guidance_hypothesis in zip(intents, extractor_examples):
                    strategy_hypothesis, guidance_hypothesis = strategy_guidance_hypothesis
                    intent_posterior_context = f'The user applies the following strategy to construct the user prompt that disguises the underlying intent.\n\nStrategy: {strategy_hypothesis}\n\nUser prompt: {prompt}\n\nThe underlying intent is: '
                    intent_posterior = self.reranker.rerank(intent_hypothesis, intent_posterior_context, relative_llh=False, length_norm=True)
                    intent_posterior_list.append(intent_posterior)
                print(f'****Intent posterior list: {intent_posterior_list}')
                intent_strategy_posterior_list = [intent_posterior + rerank_item[1] for intent_posterior, rerank_item in zip(intent_posterior_list, rerank_items)]
                scores_tensor = torch.tensor(intent_strategy_posterior_list, dtype=torch.float32)
                softmax_probs = torch.nn.functional.softmax(scores_tensor, dim=0)#.tolist()
                print(f'****Softmax probabilities: {softmax_probs}')
                cls_probs = []
                no_probs = []
                for intent_hypothesis, strategy_guidance_hypothesis in zip(intents, extractor_examples):
                    strategy_hypothesis, guidance_hypothesis = strategy_guidance_hypothesis
                    yes_prob, no_prob = self.classifier.classify(prompt, intent_hypothesis, strategy_hypothesis, output_logits=True)
                    cls_probs.append(yes_prob)
                    no_probs.append(no_prob)
                print(f'****Classifier probabilities: {cls_probs}')
                cls_probs_tensor = torch.tensor(cls_probs, dtype=torch.float32)
                no_probs_tensor = torch.tensor(no_probs, dtype=torch.float32)
                # multiply cls_probs_tensor with softmax_probs element-wise
                cls_probs_tensor = cls_probs_tensor * softmax_probs
                no_probs_tensor = no_probs_tensor * softmax_probs
                # sum cls_probs_tensor over all dimensions
                cls_prob = cls_probs_tensor.sum()
                no_prob = no_probs_tensor.sum()
                summed_prob = cls_prob + no_prob
                print(f'****Classifier final probability: {cls_prob}')
                print(f'****Classifier no probability: {no_prob}')
                print(f'****Classifier summed probability: {summed_prob}')
                if cls_prob > 0.6:
                    print(f'****Classifier decision: 1')
                    response = 'I\'m sorry, but I can\'t help with that.'
                else:
                    print(f'****Classifier decision: 0')
                    sys_prompt = "You are a helpful assistant"
                    response = self.llm.generate_response(prompt, sys_prompt=sys_prompt)
                print('='*100)
                return response
            else:
                # TODO: add classifier score
                decision = self.classifier.classify(no_strategy_intent)
                print(f'****Classifier decision (no RA): {decision}')
                if decision == 1:
                    response = 'I\'m sorry, but I can\'t help with that.'
                else:
                    sys_prompt = "You are a helpful assistant"
                    response = self.llm.generate_response(prompt, sys_prompt=sys_prompt)
                print('='*100)
                return response
        else:
            extractor_examples = [('', '')]
            intent = self.extractor.extract(prompt, extractor_examples)[0]
            decision = self.classifier.classify(intent)
            print(f'****Classifier decision (no RA): {decision}')
            if decision == 1:
                response = 'I\'m sorry, but I can\'t help with that.'
            else:
                sys_prompt = "You are a helpful assistant"
                response = self.llm.generate_response(prompt, sys_prompt=sys_prompt)
            print('='*100)
            return response

    def generate_test(self, prompt):
        query_analysis = ["The user prompt is straightforward in expressing the intent.", "The user prompt clearly conveys its intent.", "The user prompt directly communicates the intent.", "The user prompt directly expresses the intent.", "The user prompt directly conveys the intent.", "The user prompt directly communicates the intent.", "The user prompt leaves no ambiguity in expressing the intent.", "The user prompt is unambiguous in its presentation of the intent.", "The user prompt presents the intent in a straightforward manner.", "The user prompt is explicit in revealing the intent."]
        retriever_scores, jb_indices = self._query_index(prompt, 'jailbreak', 50)
        retriever_scores = retriever_scores[0]
        jb_indices = jb_indices[0]
        reranker_scores = []
        reranker_contexts = []
        for jb_index in jb_indices:
            jb_example = self.jailbreak_db_source[jb_index]
            intent_example = self.jailbreak_db_seed_queries[jb_index]
            strategy_example = self.jailbreak_db_analysis[jb_index]
            if strategy_example[0] == '"':
                strategy_example = strategy_example[1:]
            if strategy_example[-1] == '"':
                strategy_example = strategy_example[:-1]
            strategy_example = strategy_example.replace("The prompt", "The user prompt")
            strategy_example = strategy_example.replace("the prompt", 'the user prompt')
            strategy_example = strategy_example.replace("This prompt", "The user prompt")
            strategy_example = strategy_example.replace("this prompt", "the user prompt")
            if strategy_example.startswith('Analysis: '):
                strategy_example = strategy_example[10:]
            incomplete_user_prompt = prompt[:-1]
            reranker_context = f'The following strategy accurately describes the patterns of the user prompt:\nStrategy: {strategy_example}\nUser prompt: '
            reranker_contexts.append(reranker_context)
            # reranker_scores.append(self.reranker.rerank(prompt, reranker_context, relative_llh=False, length_norm=True))
        reranker_scores = self.reranker.rerank(prompt, reranker_contexts, relative_llh=False, length_norm=True)
        base_reranker_context = 'User prompt: '
        base_reranker_score = self.reranker.rerank(prompt, base_reranker_context, relative_llh=False, length_norm=True)
        # base_reranker_context = f'Evaluate the user prompt: {prompt}\n'
        # print(f'****Prompt Last Token: {prompt[-1]}')
        # base_reranker_score = self.reranker.rerank("Evaluation", base_reranker_context, relative_llh=False, length_norm=True)
        pmi_list = [rrk_score - base_reranker_score for rrk_score in reranker_scores]
        rerank_items = [(jb_index, pmi) for jb_index, pmi in zip(jb_indices, pmi_list)]
        pmi_threshold = 0.0
        rerank_items = [rerank_item for rerank_item in rerank_items if rerank_item[1] > pmi_threshold]
        if len(rerank_items) > 0:
            rerank_items.sort(key=lambda x: x[1], reverse=True)
            print(f'****Count: {len(rerank_items)}')
            rerank_k = min(10, len(rerank_items))
            rerank_items = rerank_items[:rerank_k]
            extractor_examples = []
            for jb_index, pmi in rerank_items:
                strategy_example = self.jailbreak_db_analysis[jb_index]
                guidance_example = self.jailbreak_db_guidance[jb_index]
                extractor_examples.append((strategy_example, guidance_example))
            # no_strategy_example = 'The prompt is straightforward in expressing the intent without any strategies.'


            no_strategy_example = 'The user prompt is straightforward and genuine in expressing the query.'
            no_strategy_guidance = "The intent can be taken by directly summarizing the prompt."
            no_strategy_reranker_context = f'The following strategy accurately describes the user prompt:\nStrategy: {no_strategy_example}\nUser prompt: '
            no_strategy_reranker_score = self.reranker.rerank(prompt, no_strategy_reranker_context, relative_llh=False, length_norm=True)
            no_strategy_pmi = no_strategy_reranker_score - base_reranker_score
            for no_strategy_example in query_analysis:
                no_strategy_reranker_context = f'The following strategy accurately describes the user prompt:\nStrategy: {no_strategy_example}\nUser prompt: '
                new_no_strategy_reranker_score = self.reranker.rerank(prompt, no_strategy_reranker_context, relative_llh=False, length_norm=True)
                new_no_strategy_pmi = new_no_strategy_reranker_score - base_reranker_score
                if new_no_strategy_pmi > no_strategy_pmi:
                    no_strategy_pmi = new_no_strategy_pmi
                    no_strategy_example = no_strategy_example
            print(f'****No strategy example: {no_strategy_example}')

            rerank_items.append((-1, no_strategy_pmi))
            print(f'****Rerank items: {rerank_items}')
            extractor_examples.append((no_strategy_example, no_strategy_guidance))
            intents_raw, selection = self.extractor.extract(prompt, extractor_examples, prompt_v18=True, return_selection=True)
            intents = []
            for intent_idx, intent_raw in enumerate(intents_raw):
                if intent_raw.startswith(': \n'):
                    intent_raw = intent_raw[3:].strip()
                if intent_raw.startswith(': '):
                    intent_raw = intent_raw[2:].strip()
                intents.append(intent_raw)
                print(f'****Extracted intent {intent_idx}: {intent_raw}')
            print(f'****Selection: {selection}')
            no_strategy_intent = intents[-1]
            intents = [intent for intent, sel in zip(intents[:-1], selection[:-1]) if sel == 1]
            if len(intents) > 0:
                intents.append(no_strategy_intent)
                extractor_examples = [extractor_example for extractor_example, sel in zip(extractor_examples[:-1], selection[:-1]) if sel == 1]
                extractor_examples.append((no_strategy_example, no_strategy_guidance))
                rerank_items = [rerank_item for rerank_item, sel in zip(rerank_items[:-1], selection[:-1]) if sel == 1]
                rerank_items.append((-1, no_strategy_pmi))
                intent_posterior_list = []
                intent_posterior_contexts = []
                for intent_hypothesis, strategy_guidance_hypothesis in zip(intents, extractor_examples):
                    strategy_hypothesis, guidance_hypothesis = strategy_guidance_hypothesis
                    intent_posterior_context = f'The user applies the following strategy to construct the user prompt that disguises the underlying intent.\n\nStrategy: {strategy_hypothesis}\n\nUser prompt: {prompt}\n\nThe underlying intent is: '
                    # intent_posterior_context = f'The user applies the following strategy to construct the user prompt with an underlying intent.\n\nStrategy: {strategy_hypothesis}\n\nUser prompt: {prompt}\n\nThe underlying intent is: '
                    # intent_posterior = self.reranker.rerank(intent_hypothesis, intent_posterior_context, relative_llh=False, length_norm=True)
                    # intent_posterior_list.append(intent_posterior)
                    intent_posterior_contexts.append(intent_posterior_context)
                intent_posterior_list = self.reranker.rerank(intents, intent_posterior_contexts, relative_llh=False, length_norm=True)
                print(f'****Intent posterior list: {intent_posterior_list}')
                # strategy_posterior_list = []
                # for strategy_hypothesis, guidance_hypothesis in extractor_examples:
                #     strategy_posterior_context = f'User prompt: {prompt}\n\nThe following strategy accurately describes how the user prompt was constructed.\n\nStrategy: '
                #     strategy_posterior = self.reranker.rerank(strategy_hypothesis, strategy_posterior_context, relative_llh=False, length_norm=True)
                #     strategy_posterior_list.append(strategy_posterior)
                # print(f'****Strategy posterior list: {strategy_posterior_list}')
                intent_strategy_posterior_list = [intent_posterior + rerank_item[1] for intent_posterior, rerank_item in zip(intent_posterior_list, rerank_items)]
                # intent_strategy_posterior_list = [intent_posterior + strategy_posterior for intent_posterior, strategy_posterior in zip(intent_posterior_list, strategy_posterior_list)]
                scores_tensor = torch.tensor(intent_strategy_posterior_list, dtype=torch.float32)
                print(f'****Scores tensor: {scores_tensor}')
                softmax_probs = torch.nn.functional.softmax(scores_tensor, dim=0)#.tolist()
                print(f'****Softmax probabilities: {softmax_probs}')
                cls_probs = []
                no_probs = []
                for intent_hypothesis, strategy_guidance_hypothesis in zip(intents, extractor_examples):
                    strategy_hypothesis, guidance_hypothesis = strategy_guidance_hypothesis
                    yes_prob, no_prob = self.classifier.classify(prompt, intent_hypothesis, strategy_hypothesis, output_logprob=True)
                    cls_probs.append(yes_prob)
                    no_probs.append(no_prob)
                print(f'****Classifier probabilities: {cls_probs}')
                cls_probs_tensor = torch.tensor(cls_probs, dtype=torch.float32)
                no_probs_tensor = torch.tensor(no_probs, dtype=torch.float32)
                # multiply cls_probs_tensor with softmax_probs element-wise
                cls_probs_tensor = cls_probs_tensor * softmax_probs
                no_probs_tensor = no_probs_tensor * softmax_probs
                # sum cls_probs_tensor over all dimensions
                cls_prob = cls_probs_tensor.sum()
                no_prob = no_probs_tensor.sum()
                summed_prob = cls_prob + no_prob
                print(f'****Classifier final probability: {cls_prob}')
                print(f'****Classifier no probability: {no_prob}')
                print(f'****Classifier summed probability: {summed_prob}')
                if cls_prob > 0.6:
                    print(f'****Classifier decision: 1')
                    response = 'I\'m sorry, but I can\'t help with that.'
                else:
                    print(f'****Classifier decision: 0')
                    sys_prompt = "You are a helpful assistant"
                    response = self.llm.generate_response(prompt, sys_prompt=sys_prompt)
                print('='*100)
                return response
            else:
                # TODO: add classifier score
                decision = self.classifier.classify(no_strategy_intent)
                print(f'****Classifier decision (no RA): {decision}')
                if decision == 1:
                    response = 'I\'m sorry, but I can\'t help with that.'
                else:
                    sys_prompt = "You are a helpful assistant"
                    response = self.llm.generate_response(prompt, sys_prompt=sys_prompt)
                print('='*100)
                return response
        else:
            extractor_examples = [('', '')]
            intent = self.extractor.extract(prompt, extractor_examples)[0]
            decision = self.classifier.classify(intent)
            print(f'****Classifier decision (no RA): {decision}')
            if decision == 1:
                response = 'I\'m sorry, but I can\'t help with that.'
            else:
                sys_prompt = "You are a helpful assistant"
                response = self.llm.generate_response(prompt, sys_prompt=sys_prompt)
            print('='*100)
            return response
