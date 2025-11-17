import os
# tokenizers 병렬 처리 경고 방지 (프로세스 fork 전에 설정해야 함)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import time
import shutil
from torch.utils.data import Dataset
import torch.distributed as dist

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


class SemanticTextSplitter:
    """
    문서 구조 기반 semantic text splitter
    Chapter, Section, Subsection 단위로 문서를 분할
    """
    
    def __init__(self, chunk_size=1000, chunk_overlap=200, min_chunk_size=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # 문서 구조 패턴 정의
        self.patterns = {
            'chapter': re.compile(r'^Chapter\s+(\d+(?:\.\d+)?)\s+(.+)$', re.MULTILINE | re.IGNORECASE),
            'section': re.compile(r'^(\d+\.\d+)\s+(.+)$', re.MULTILINE),
            'subsection': re.compile(r'^(\d+\.\d+\.\d+)\s+(.+)$', re.MULTILINE),
            'table': re.compile(r'^Table\s+(\d+(?:\.\d+)?)\s+(.+)$', re.MULTILINE | re.IGNORECASE),
            'figure': re.compile(r'^Figure\s+(\d+(?:\.\d+)?)\s+(.+)$', re.MULTILINE | re.IGNORECASE),
        }
    
    def _parse_structure(self, text):
        """문서 구조 파싱 - Chapter, Section, Subsection 위치 찾기"""
        structure = []
        lines = text.split('\n')
        
        current_chapter = None
        current_section = None
        current_subsection = None
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Chapter 검사
            chapter_match = self.patterns['chapter'].match(line_stripped)
            if chapter_match:
                current_chapter = {
                    'type': 'chapter',
                    'number': chapter_match.group(1),
                    'title': chapter_match.group(2).strip(),
                    'line': i,
                    'content_start': i
                }
                current_section = None
                current_subsection = None
                structure.append(current_chapter)
                continue
            
            # Section 검사
            section_match = self.patterns['section'].match(line_stripped)
            if section_match:
                current_section = {
                    'type': 'section',
                    'number': section_match.group(1),
                    'title': section_match.group(2).strip(),
                    'line': i,
                    'content_start': i,
                    'chapter': current_chapter['number'] if current_chapter else None
                }
                current_subsection = None
                structure.append(current_section)
                continue
            
            # Subsection 검사
            subsection_match = self.patterns['subsection'].match(line_stripped)
            if subsection_match:
                current_subsection = {
                    'type': 'subsection',
                    'number': subsection_match.group(1),
                    'title': subsection_match.group(2).strip(),
                    'line': i,
                    'content_start': i,
                    'chapter': current_chapter['number'] if current_chapter else None,
                    'section': current_section['number'] if current_section else None
                }
                structure.append(current_subsection)
                continue
            
            # Table 검사
            table_match = self.patterns['table'].match(line_stripped)
            if table_match:
                structure.append({
                    'type': 'table',
                    'number': table_match.group(1),
                    'title': table_match.group(2).strip(),
                    'line': i,
                    'content_start': i,
                    'chapter': current_chapter['number'] if current_chapter else None,
                    'section': current_section['number'] if current_section else None
                })
                continue
            
            # Figure 검사
            figure_match = self.patterns['figure'].match(line_stripped)
            if figure_match:
                structure.append({
                    'type': 'figure',
                    'number': figure_match.group(1),
                    'title': figure_match.group(2).strip(),
                    'line': i,
                    'content_start': i,
                    'chapter': current_chapter['number'] if current_chapter else None,
                    'section': current_section['number'] if current_section else None
                })
                continue
        
        return structure
    
    def split_documents(self, documents):
        """Document 리스트를 구조 기반으로 분할"""
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            text = doc.page_content
            source = doc.metadata.get('source', 'unknown')
            
            # 문서 구조 파싱
            structure = self._parse_structure(text)
            lines = text.split('\n')
            
            if not structure:
                # 구조가 없으면 RecursiveCharacterTextSplitter 사용
                fallback_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                chunks = fallback_splitter.split_documents([doc])
                for chunk_idx, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        'chunk_index': chunk_idx,
                        'chapter': None,
                        'section': None,
                        'content_type': 'text',
                        'char_start': text.find(chunk.page_content),
                        'char_end': text.find(chunk.page_content) + len(chunk.page_content)
                    })
                all_chunks.extend(chunks)
                continue
            
            # 구조 기반 분할
            for struct_idx, struct in enumerate(structure):
                # 현재 구조 요소의 시작 라인
                start_line = struct['line']
                
                # 다음 구조 요소의 시작 라인 (또는 문서 끝)
                if struct_idx + 1 < len(structure):
                    end_line = structure[struct_idx + 1]['line']
                else:
                    end_line = len(lines)
                
                # 해당 구간의 텍스트 추출
                section_lines = lines[start_line:end_line]
                section_text = '\n'.join(section_lines).strip()
                
                if len(section_text) < self.min_chunk_size:
                    continue
                
                # 청크가 너무 크면 RecursiveCharacterTextSplitter로 재분할
                if len(section_text) > self.chunk_size:
                    temp_doc = Document(
                        page_content=section_text,
                        metadata={'source': source}
                    )
                    fallback_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap
                    )
                    sub_chunks = fallback_splitter.split_documents([temp_doc])
                    
                    for sub_idx, sub_chunk in enumerate(sub_chunks):
                        char_start = text.find(sub_chunk.page_content)
                        sub_chunk.metadata.update({
                            'chunk_index': len(all_chunks) + sub_idx,
                            'chapter': struct.get('chapter'),
                            'section': struct.get('section'),
                            'subsection': struct.get('number') if struct['type'] == 'subsection' else None,
                            'content_type': struct['type'],
                            'struct_number': struct.get('number'),
                            'struct_title': struct.get('title'),
                            'char_start': char_start if char_start >= 0 else 0,
                            'char_end': char_start + len(sub_chunk.page_content) if char_start >= 0 else len(sub_chunk.page_content)
                        })
                    all_chunks.extend(sub_chunks)
                else:
                    # 청크 크기가 적절하면 그대로 사용
                    char_start = text.find(section_text)
                    chunk = Document(
                        page_content=section_text,
                        metadata={
                            'source': source,
                            'chunk_index': len(all_chunks),
                            'chapter': struct.get('chapter'),
                            'section': struct.get('section'),
                            'subsection': struct.get('number') if struct['type'] == 'subsection' else None,
                            'content_type': struct['type'],
                            'struct_number': struct.get('number'),
                            'struct_title': struct.get('title'),
                            'char_start': char_start if char_start >= 0 else 0,
                            'char_end': char_start + len(section_text) if char_start >= 0 else len(section_text)
                        }
                    )
                    all_chunks.append(chunk)
        
        return all_chunks


class Planner:
    def __init__(self,
                tokenizer,
                llm,
                retriever,
                max_tokens,
                device
                ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.max_tokens = max_tokens
        self.device = device
        
        self.target_labels = "normal(healthy), misalignment, looseness, unbalance, bearing fault"
        
    def retrieve(self, current_knowledge):
        """
        큰 변화율과 비정상 지표에 집중한 검색 쿼리 생성 및 문서 검색
        
        Args:
            current_knowledge: 정상 상태 대비 변화율(%) 딕셔너리
                - 형식: {'rms_x': 3622.7905, 'kurtosis_x': -71.4348, ...}
                - 양수: 정상보다 증가 (예: 316.26% = 정상 대비 316% 증가)
                - 음수: 정상보다 감소 (예: -75.28% = 정상 대비 75% 감소)
                - 변화율 = (현재값 - 정상값) / 정상값 * 100
                - 0이 아닌 값만 포함됨 (feature_extract.py에서 필터링됨)
        """
        # 1. 변화율 분류 함수
        def classify_changes(change_rates, threshold_extreme=50.0, threshold_large=20.0, threshold_moderate=10.0):
            """
            변화율을 극단/큰/중간으로 분류하고 절댓값 기준 내림차순 정렬
            Returns: (extreme_features, large_features, moderate_features)
            """
            sorted_changes = sorted(
                change_rates.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            extreme_features = []  # 극단적 변화 (>=50%)
            large_features = []     # 큰 변화 (20-50%)
            moderate_features = []  # 중간 변화 (10-20%)
            
            for feature, change_rate in sorted_changes:
                abs_rate = abs(change_rate)
                if abs_rate >= threshold_extreme:
                    extreme_features.append((feature, change_rate))
                elif abs_rate >= threshold_large:
                    large_features.append((feature, change_rate))
                elif abs_rate >= threshold_moderate:
                    moderate_features.append((feature, change_rate))
            
            return extreme_features, large_features, moderate_features
        
        # 변화율 직접 사용 (이미 딕셔너리 형식이고 필터링됨)
        change_rates = current_knowledge if isinstance(current_knowledge, dict) else {}
        
        # 변화율 분류
        extreme_features, large_features, moderate_features = classify_changes(change_rates)
        
        # 3. 고장 유형별 중요 특징 매핑 테이블
        fault_critical_features = {
            "misalignment": {
                "primary": ["order_x_2x", "order_y_2x", "order_x_1x", "order_y_1x"],
                "secondary": ["rms_x", "rms_y", "peak_freq_x", "peak_freq_y", "peak2peak_x", "peak2peak_y"]
            },
            "unbalance": {
                "primary": ["order_x_1x", "order_y_1x", "rms_x", "rms_y"],
                "secondary": ["peak_freq_x", "peak_freq_y", "peak2peak_x", "peak2peak_y", "peak_abs_x", "peak_abs_y"]
            },
            "looseness": {
                "primary": ["kurtosis_x", "kurtosis_y", "crest_factor_x", "crest_factor_y", 
                           "order_x_3x", "order_y_3x"],
                "secondary": ["skewness_x", "skewness_y", "peak2peak_x", "peak2peak_y", "var_x", "var_y"]
            },
            "bearing fault": {
                "primary": ["order_x_2x", "order_x_3x", "order_y_2x", "order_y_3x",
                           "peak_freq_x", "peak_freq_y"],
                "secondary": ["rms_freq_x", "rms_freq_y", "center_freq_x", "center_freq_y", 
                             "bpfo_peak_x", "bpfi_peak_x"]
            }
        }
        
        # 4. 가중치 기반 고장 유형 추론
        fault_scores = {}
        all_abnormal_features = extreme_features + large_features
        
        for fault_type, features in fault_critical_features.items():
            score = 0.0
            # Primary 특징에 더 높은 가중치 (2.0)
            for feat_name, change_rate in all_abnormal_features:
                if feat_name in features["primary"]:
                    score += abs(change_rate) * 2.0
                elif feat_name in features["secondary"]:
                    score += abs(change_rate) * 1.0
            
            if score > 0:
                fault_scores[fault_type] = score
        
        # 점수가 높은 상위 2개 고장 유형 선택
        suspected_faults = []
        if fault_scores:
            sorted_faults = sorted(fault_scores.items(), key=lambda x: x[1], reverse=True)
            suspected_faults = [fault for fault, score in sorted_faults[:2]]
        
        # 고장 유형별 키워드 매핑
        fault_keywords = {
            "misalignment": ["2x harmonic", "second harmonic", "misalignment", "axial", "radial"],
            "unbalance": ["1x harmonic", "first harmonic", "unbalance", "rotational"],
            "looseness": ["3x harmonic", "looseness", "impact", "shock", "kurtosis", "crest factor"],
            "bearing fault": ["bearing", "BPFO", "BPFI", "ball pass", "inner race", "outer race"],
        }

        print(f"suspected_faults: {suspected_faults}")
        print(f"fault_keywords: {fault_keywords[suspected_faults[0]]}")
        print(f"fault_keywords: {fault_keywords[suspected_faults[1]]}")
        
        # 5. 검색 쿼리 생성 (비정상 특징 강조)
        query_parts = [
            "vibration diagnosis rotating machinery",
            f"diagnostic methods classify {self.target_labels}",
        ]
        
        # 극단적 변화 특징 강조
        if extreme_features:
            extreme_feat_names = [f[0] for f in extreme_features[:5]]  # 상위 5개
            query_parts.append("CRITICAL abnormal features")
            query_parts.extend(extreme_feat_names)
            query_parts.append("extreme deviation from normal")
        
        # 큰 변화 특징 포함
        if large_features:
            large_feat_names = [f[0] for f in large_features[:5]]
            query_parts.append("significant changes")
            query_parts.extend(large_feat_names)
        
        # 추론된 고장 유형 키워드 추가
        if suspected_faults:
            for fault in suspected_faults:
                query_parts.extend(fault_keywords.get(fault, []))
        else:
            # 모든 고장 유형 키워드 포함 (다양성 확보)
            for keywords in fault_keywords.values():
                query_parts.extend(keywords[:2])
        
        # 특징 기반 키워드 추가
        query_parts.extend([
            "order spectrum analysis",
            "RMS vibration",
            "kurtosis",
            "crest factor",
            "frequency domain",
            "harmonic components",
            "ISO 7919",
            "ISO 10816",
            "diagnostic thresholds"
        ])
        
        # 최종 쿼리 구성
        base_query = " ".join(query_parts)
        
        # 비정상 특징 요약 생성
        abnormal_summary = []
        if extreme_features:
            extreme_summary = ", ".join([f"{f[0]}={f[1]:.1f}%" for f in extreme_features[:3]])
            abnormal_summary.append(f"EXTREME changes (>50%): {extreme_summary}")
        if large_features:
            large_summary = ", ".join([f"{f[0]}={f[1]:.1f}%" for f in large_features[:3]])
            abnormal_summary.append(f"LARGE changes (20-50%): {large_summary}")
        
        # current_knowledge를 문자열로 변환 (표시용)
        if isinstance(current_knowledge, dict):
            knowledge_str = ", ".join([f"{k}: {v:.4f}" for k, v in list(current_knowledge.items())[:20]])
        else:
            knowledge_str = str(current_knowledge)[:400]
        
        query = (
            f"{base_query}. "
            f"Focus on abnormal indicators with significant deviation from normal baseline. "
            f"{' '.join(abnormal_summary)}. "
            f"Current vibration state (change rates % from normal baseline): {knowledge_str}. "
            "Note: Change rates show percentage deviation from normal state. "
            "Positive values indicate increase from normal, negative values indicate decrease. "
            "For example, order_x_2x: 316% means the 2nd harmonic component is 316% higher than normal. "
            "Provide diagnostic criteria, thresholds, and classification methods specifically for these abnormal features."
        )
        
        retrieved_docs = self.retriever.invoke(query)
        
        return retrieved_docs
    
    def plan(self, current_knowledge, retrieve_docs):
        """
        현재 상태를 분석하고 구체적인 진단을 수행
        
        Args:
            current_knowledge: 정상 상태 대비 변화율(%) 딕셔너리
                - 형식: {'rms_x': 3622.7905, 'kurtosis_x': -71.4348, ...}
                - 양수: 정상보다 증가 (예: 316.26% = 정상 대비 316% 증가)
                - 음수: 정상보다 감소 (예: -75.28% = 정상 대비 75% 감소)
                - 변화율 = (현재값 - 정상값) / 정상값 * 100
                - 0이 아닌 값만 포함됨 (feature_extract.py에서 필터링됨)
            retrieve_docs: 검색된 문서들
        """
        # current_knowledge를 문자열로 변환
        if isinstance(current_knowledge, dict):
            knowledge_str = "\n".join([f"{k}: {v:.4f}" for k, v in current_knowledge.items()])
        else:
            knowledge_str = str(current_knowledge)
        
        prompt = (
            "System: You are a senior vibration analyst. Analyze the CURRENT STATE and provide SPECIFIC DIAGNOSIS based on actual change rate values. Be precise and cite sources. Answer in korean.\n"
            f"User: Diagnose the CURRENT STATE of rotating machinery among: {self.target_labels}.\n\n"
            "CURRENT STATE ANALYSIS DATA:\n"
            f"{knowledge_str}\n\n"
            "DATA FORMAT EXPLANATION:\n"
            "- Each line shows a feature name and its change rate percentage from normal baseline.\n"
            "- Format: \"feature_name: change_rate%\"\n"
            "- Positive values (e.g., 316.26%): The feature value is HIGHER than normal (increased by 316%)\n"
            "- Negative values (e.g., -75.28%): The feature value is LOWER than normal (decreased by 75%)\n"
            "- Change rate = (current_value - normal_value) / normal_value * 100\n"
            "- Example: \"order_x_2x: 316.26%\" means the 2nd harmonic component (order_x_2x) is 316% higher than the normal baseline.\n"
            "- Example: \"crest_factor_x: -75.28%\" means the crest factor is 75% lower than normal.\n\n"
            "Evidence snippets from manuals/papers (each prefixed with [DOC#]):\n"
            f"{retrieve_docs}\n\n"
            "TASK:\n"
            "1. CURRENT STATE ANALYSIS: Analyze the actual change rate values. Identify which features show extreme/large deviations (>50% or 20-50%) and explain what they indicate.\n"
            "   - Focus on features with absolute change rate >= 50% (extreme) or >= 20% (large)\n"
            "   - Explain what each abnormal feature indicates (e.g., 'order_x_2x increased by 316% indicates strong 2nd harmonic, typical of misalignment')\n"
            "2. SPECIFIC DIAGNOSIS CONCLUSION: Based on the actual change rate values, determine the most likely fault type(s) with confidence level and specific reasoning.\n"
            "   - Use the actual percentage values in your reasoning (e.g., 'order_x_2x is 316% higher than normal')\n"
            "   - Cite specific values and compare them with diagnostic criteria from evidence\n"
            "3. DIAGNOSIS PLAN: Provide general diagnostic guidelines for each fault type.\n\n"
            "Return STRICT JSON with keys: current_analysis, diagnosis_conclusion, diagnosis_plan.\n\n"
            "- current_analysis: {\n"
            "    \"abnormal_features\": [\n"
            "      {\"feature\": \"order_x_2x\", \"change_rate\": 316.26, \"category\": \"extreme\", \"interpretation\": \"2차 조화 성분이 정상 대비 316% 증가하여 축 불일치 징후\"}\n"
            "    ],\n"
            "    \"summary\": \"현재 상태 요약: 주요 비정상 특징과 의미\"\n"
            "  }\n\n"
            "- diagnosis_conclusion: {\n"
            "    \"most_likely_fault\": \"misalignment\",\n"
            "    \"confidence\": \"high\",\n"
            "    \"reasoning\": \"order_x_2x가 정상 대비 316%, order_y_2x가 정상 대비 364% 증가하여 2차 조화 성분이 강하게 나타남. 이는 ISO 13373-1에 따른 축 불일치의 전형적 징후 [DOC4].\",\n"
            "    \"supporting_features\": [\"order_x_2x\", \"order_y_2x\"],\n"
            "    \"alternative_faults\": [{\"fault\": \"looseness\", \"reason\": \"crest_factor_x가 정상 대비 -69% 감소\"}]\n"
            "  }\n\n"
            f"- diagnosis_plan: object with keys {self.target_labels}, each a list of diagnosis rules;\n"
            "  every rule item must be a JSON object: {\"diagnosis idea\": <one line>, \"why\": <short reason>, \"source\": \"DOC#\"}.\n\n"
            "IMPORTANT:\n"
            "- Use ACTUAL CHANGE RATE VALUES from CURRENT STATE in your analysis (e.g., 'order_x_2x is 316% higher than normal' not 'order_x_2x is high').\n"
            "- Always mention '정상 대비 X% 증가/감소' when referring to change rates.\n"
            "- Be SPECIFIC about the current state using actual percentage values, not generic guidelines.\n"
            "- Cite sources from evidence snippets using [DOC#] format.\n"
            "- All text must be in korean.\n"
            "Assistant: Output JSON only, no extra text."
        )
        
        # vLLM 사용 여부 확인 (hasattr로 체크)
        if hasattr(self.llm, 'client'):  # VLLMWrapper인 경우
            # vLLM은 텍스트를 직접 받음
            out_ids = self.llm.generate(
                input_ids=None,  # 사용하지 않음
                max_new_tokens=self.max_tokens,
                do_sample=False,
                prompt=prompt  # prompt 직접 전달
            )
            out_text = out_ids[0][0] if isinstance(out_ids[0], list) else out_ids[0]
        else:
            # HuggingFace 모델 사용
            with torch.no_grad():
                input_ids = self.tokenizer(prompt, return_tensors='pt')
                # BatchEncoding을 디바이스로 이동 (딕셔너리처럼 처리)
                if hasattr(input_ids, 'to'): # transformers 4.20.0 이상 버전에서 작동
                    input_ids = input_ids.to(self.device)
                else: # transformers 구버전에서 작동
                    input_ids = {k: v.to(self.device) for k, v in input_ids.items()}
                out_ids = self.llm.generate(**input_ids, max_new_tokens=self.max_tokens, do_sample=False)
                out_text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        
        return out_text.strip()
    
    def summarize(self, plan):
        """
        진단 계획을 요약
        
        Args:
            plan: 진단 계획 텍스트
        """
        prompt = (
            "System: You are a senior vibration analyst. Create a compact briefing from the given thinking/plan.\n"
            "User: Summarize given PLAN into STRICT JSON with keys: current_analysis, diagnosis_conclusion, diagnosis_plan.\n"
            "PLAN:\n" + plan + "\n"
            "Assistant: Output JSON only."
        )
        
        # vLLM 사용 여부 확인
        if hasattr(self.llm, 'client'):  # VLLMWrapper인 경우
            out_ids = self.llm.generate(
                input_ids=None,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                prompt=prompt
            )
            out_text = out_ids[0][0] if isinstance(out_ids[0], list) else out_ids[0]
        else:
            # HuggingFace 모델 사용
            with torch.no_grad():
                input_ids = self.tokenizer(prompt, return_tensors='pt')
                # BatchEncoding을 디바이스로 이동 (딕셔너리처럼 처리)
                if hasattr(input_ids, 'to'):
                    input_ids = input_ids.to(self.device)
                else:
                    input_ids = {k: v.to(self.device) for k, v in input_ids.items()}
                out_ids = self.llm.generate(**input_ids, max_new_tokens=self.max_tokens, do_sample=False)
                out_text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        
        return out_text.strip()
    
    def __call__(self, current_knowledge):
        """
        현재 상태를 분석하고 진단 계획 생성
        
        Args:
            current_knowledge: 정상 상태 대비 변화율(%) 문자열
        """
        retrieve_docs = self.retrieve(current_knowledge)
        plan = self.plan(current_knowledge, retrieve_docs)
        
        evidence = retrieve_docs
        try:
            json_part = plan.split('Assistant: Output JSON only, no extra text. Do not hallucinate.')[1]
        except IndexError:
            try:
                # 다른 형식의 프롬프트도 시도
                json_part = plan.split('Assistant: Output JSON only, no extra text.')[1]
            except IndexError:
                json_part = plan  # 전체를 반환
        
        return json_part

class VibrationTokenizer(nn.Module):
    def __init__(self, vib_encoder, token_embed_dim, freeze_encoder=True, embedding_dim=768):
        super().__init__()
        self.vib_encoder = vib_encoder
        self.device = next(self.vib_encoder.parameters()).device
        self.dtype = next(self.vib_encoder.parameters()).dtype

        if freeze_encoder:
            for param in self.vib_encoder.parameters():
                param.requires_grad = False

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=embedding_dim,
                out_features=int(embedding_dim*2)
            ),
            nn.Sigmoid(),
            nn.Linear(
                in_features=int(embedding_dim*2),
                out_features=token_embed_dim
            )
        )

    def forward(self, x):
        # Ensure inputs are on the same device as the encoder/model
        device = next(self.vib_encoder.parameters()).device if self.vib_encoder is not None else next(self.model.parameters()).device

        current_tensor = x.unsqueeze(0).to(device)

        sample_attn = self.vib_encoder._encode_full(current_tensor)
        sample_attn = sample_attn[:, 0, :]  # 768

        z = self.model(sample_attn)

        # Return CPU tensors so DataLoader pin_memory works properly
        return z.detach().cpu()

class VibrationSFTDataset(Dataset):
    def __init__(self,
                vibration_dataset,
                vib_tokenizer,
                planner,
                device,
                test_mode=True):
        self.vibration_dataset = vibration_dataset
        self.vib_tokenizer = vib_tokenizer
        self.planner = planner
        self.target_labels = "normal(healthy), misalignment, looseness, unbalance, bearing fault"
        self.device=device 
        self.test_mode = test_mode
        
    def __len__(self):
        return len(self.vibration_dataset)
    
    def __getitem__(self, idx):
        (
            current_x, _, current_info, normal_x, _, normal_info
        ) = self.vibration_dataset.__getitem__(idx, data_info=True)
        # Keep tensors on CPU; tokenizer will move to correct device internally

        current_knowledge = current_info['knowledge']
        # normal_knowledge는 더 이상 사용하지 않음 (current_knowledge가 이미 변화율을 포함)
        
        current_token, normal_token = self.vib_tokenizer(current_x, normal_x)
        current_token = current_token.squeeze(0)
        normal_token = normal_token.squeeze(0)
        
        if self.test_mode:
            print('Skip Plan')
            plan_text = 'See you Later'
        else:
            plan_text = self.planner(current_knowledge)
        
        system_prompt = (
            f"""You are a domain expert in vibration-based rotating machinery diagnostics.
            Make precise diagnosis classification among {self.target_labels} from User question.
            """
        )
        print(plan_text)
        # 요구사항: reasoning(접근 계획/추론)과 result(최종 JSON) 분리 출력
        # - reasoning: 단계적으로 사고 과정을 요약 (Step 1, Step 2 ...)
        # - answer: 기존 JSON 형식 하나만 출력
        user_prompt = (
            f"Possible conditions: {self.target_labels}.\n\n"
            "You will perform a 3-stage diagnostic reasoning process comparing NORMAL vs CURRENT states.\n"
            "Use ONLY the two vibration embeddings when doing vibration-only reasoning"
            "Use provided knowledge when doing knowledge-only reasoning.\n"
            "Do not include any code fences. Output exactly one reasoning block and one answer block.\n\n"
            "Vibration embeddings:\n"
            "- Normal state embedding: <NORMAL_VIB_EMB>\n"
            "- Current state embedding: <CURRENT_VIB_EMB>\n\n"
            f"- Current state features (change rates % from normal): {current_knowledge}\n"
            "Note: Change rates show percentage deviation from normal baseline. Positive values indicate increase, negative values indicate decrease.\n"
            f"Diagnosis plan :\n{plan_text}\n\n"
            "Follow the EXACT structure below. Write detailed analysis in <reasoning> and concise result in <answer>."
            """
            <reasoning>
            MAIN STEP 1 — Embedding-based Comparison
            1.1 Compute or conceptually assess similarity/distance between <CURRENT_VIB_EMB> and <NORMAL_VIB_EMB> (e.g., cosine proximity). 
            1.2 Identify which label prototypes the CURRENT embedding is closest to, if implied by features/notes.
            1.3 Summarize whether embeddings alone suggest normality or a specific fault class, and why.

            MAIN STEP 2 — Feature- & Knowledge-based Classification
            2.1 Compare CURRENT vs NORMAL for RMS, kurtosis, crest factor per axis. Note significant deviations.
            2.2 Inspect spectral content in orders: 
                - Check prominence at 1×, 2×, 3× and higher-order harmonics.
                - Note sidebands, modulations, or resonance clusters.
                - Map observed peaks to bearing fault frequencies (BPFO/BPFI/BSF/FTF) when provided.
            2.3 Cross-reference CurrentKnowledgeSnippets and NormalKnowledgeSnippets with observed stats/spectrum:
                - Confirm or refute each snippet with concrete evidence.
            2.4 Derive a label from features+knowledge with a rationale and uncertainty.

            MAIN STEP 3 — Fused Decision
            3.1 Reconcile STEP 1 and STEP 2 outcomes. If they agree, keep the label; if not, pick the label with stronger evidence and explain why.
            3.2 State key indicators (top 2–3) that most strongly support the final label.
            3.3 Provide a calibrated confidence (0–1), reflecting agreement between steps, signal quality, and evidence strength.
            </reasoning>
            """
            "<answer>{\n"
            "  \"vib_only_label\": <one_of_labels>,\n"
            "  \"vib_reason\": <one_sentence>,\n"
            "  \"knowledge_only_label\": <one_of_labels>,\n"
            "  \"knowledge_reason\": <one_sentence>,\n"
            "  \"criteria\": <one_or_two_bullets>,\n"
            "  \"final_label\": <one_of_labels>,\n"
            "  \"fusion_reason\": <one_sentence>\n"
            "}</answer>\n"
            "Constraints: Only one reasoning block and one answer block. No extra text."
        )

        prompt_only = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
        
        assistant_response = f"The diagnosis result is {current_info['label_class']}."

        return {
            'prompt': prompt_only,
            'answers': assistant_response,
            'normal_token': normal_token,
            'current_token': current_token
        }

def make_retriever(embedding_model, model_cache, docs_path, retriever_k, rebuild=False):
    """
    벡터 스토어와 retriever 생성
    
    Args:
        embedding_model: 임베딩 모델명
        model_cache: 모델 캐시 경로
        docs_path: 문서 경로
        retriever_k: 검색할 문서 수
        rebuild: True면 기존 벡터 스토어 삭제 후 재생성, False면 기존 것 재사용
    """
    persist_directory = os.path.join(docs_path, "vectorstore")
    
    # 임베딩 모델 로딩 (재사용 시에도 필요)
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=model_cache
    )
    
    # 기존 벡터 스토어가 있고 rebuild=False면 재사용
    if os.path.exists(persist_directory) and not rebuild:
        print(f"기존 벡터 스토어 로드 중: {persist_directory}")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        print(f"벡터 스토어 로드 완료 (기존 DB 사용)")
    else:
        # 기존 벡터 스토어 삭제 (rebuild=True이거나 기존 DB가 없는 경우)
        if os.path.exists(persist_directory):
            print(f"기존 벡터 스토어 삭제 중: {persist_directory}")
            shutil.rmtree(persist_directory)
        
        # 2. docs_path 폴더에 있는 TXT 파일들을 불러오기
        txt_files = [os.path.join(docs_path, f) for f in os.listdir(docs_path) if f.lower().endswith('.txt')]
        if not txt_files:
            raise ValueError(f"docs_path에 .txt 파일이 없습니다: {docs_path}")
        
        raw_docs = []
        for path in txt_files:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                raw_docs.append(Document(page_content=text,
                                metadata={'source': os.path.basename(path)}))

        print(f"로드된 문서 수: {len(raw_docs)}")

        # 3. SemanticTextSplitter로 문서 구조 기반 분할 (메타데이터 강화 포함)
        text_splitter = SemanticTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(raw_docs)
        print(f"생성된 청크 수: {len(docs)}")

        # 4. VectorDB에 문서 저장
        print("벡터 스토어 생성 중...")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print(f"벡터 스토어 저장 완료: {persist_directory}")
                
    # 5. MMR 검색을 사용하여 Retriever 생성 (검색 다양성 향상)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": retriever_k,
            "fetch_k": retriever_k * 3,  # 다양성 확보를 위해 더 많은 후보 검색
            "lambda_mult": 0.5  # 유사도와 다양성 균형 (0.0=다양성, 1.0=유사도)
        }
    )
    
    return retriever


class LLMDataset_Cache(torch.utils.data.Dataset):
    """
    캐시를 원본 튜플 형식으로 다시 내보내는 래퍼
    (current_x, _, current_info, normal_x, _, normal_info, plan_text)
    """
    def __init__(self, cache_blob_path, using_dataset=['iis']):
        cache_blob = torch.load(cache_blob_path)
        all_records = cache_blob["records"]
        all_datasets = cache_blob["dataset"]
        # using_dataset 안에 포함된 dataset만 남김
        filtered_records = []
        for rec, ds_name in zip(all_records, all_datasets):
            if ds_name in using_dataset:
                filtered_records.append(rec)
        self.records = filtered_records
        self.dataset_names = using_dataset

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        current_x = r["current_x"]
        current_info = r["current_info"]
        normal_x = r["normal_x"]
        normal_info = r["normal_info"]
        plan_text = r["plan_text"]
        
        current_knowledge = current_info['knowledge']
        normal_knowledge = normal_info['knowledge']
        
        system_prompt = (
            "You are a world-class AI diagnostic engineer specializing in the vibration analysis of rotating machinery. "
            "Your mission is to meticulously analyze vibration data and associated physical knowledge to deliver a precise and well-supported diagnosis. "
            "You must follow a structured, multi-stage reasoning process and present your findings in the exact format required."
        )
        user_prompt = f"""
        ### **TASK DESCRIPTION**
        Your task is to diagnose the current state of a rotating machine. Based on the provided data, you must classify the state into one of the following five categories: **normal(healthy), misalignment, looseness, unbalance, bearing fault**.

        ### **PROVIDED DATA**

        **1. Vibration Tokens (Proprietary Embeddings):**
        - Normal State Token: `<NORMAL_VIB_EMB>`
        - Current State Token: `<CURRENT_VIB_EMB>`

        **2. Physical Knowledge (Extracted Features):**
        - Current State Features (change rates % from normal baseline): "{current_knowledge}"
        "Note: Change rates show percentage deviation from normal. Positive = increase from normal, Negative = decrease from normal."

        **3. Analysis Plan & Diagnostic Criteria (Reference Guide):**
        This guide outlines the step-by-step analysis process and the criteria for each potential fault. You must use this as your primary reference for knowledge-based analysis.
        - Plan: {plan_text}

        ### **INSTRUCTIONS**

        You must perform a rigorous 3-step diagnostic reasoning process. Adhere strictly to the following structure within the `<reasoning>` block.

        <reasoning>
        **Step 1: Vibration Token-based Analysis**
        1.1. Compare the `<CURRENT_VIB_EMB>` with the `<NORMAL_VIB_EMB>`. Based on their similarity (or lack thereof), what is the initial diagnosis? A significant deviation from the normal token suggests a fault.
        1.2. Briefly state the most likely condition suggested by the vibration tokens alone.

        **Step 2: Physical Knowledge-based Analysis**
        2.1. **Feature Comparison**: Compare the 'Current State Features' against the 'Normal State Features'. Note any significant differences in RMS, Kurtosis, and Crest Factor.
        2.2. **Spectral Analysis**: Examine the 'TopPeaksHz' and 'Orders' in the current state. According to the 'Analysis Plan', do these spectral components indicate a specific fault (e.g., harmonics for unbalance/misalignment, non-harmonics for bearing faults)?
        2.3. **Criteria Matching**: Systematically check the diagnostic ideas for each fault type listed in the 'Analysis Plan'. Which condition's criteria best match the 'Current State Features'? Provide evidence.

        **Step 3: Fused Decision**
        3.1. **Synthesize Findings**: Combine the conclusions from Step 1 (token-based) and Step 2 (knowledge-based).
        3.2. **Final Diagnosis**: If they agree, confirm the diagnosis. If they conflict, determine which conclusion is supported by stronger evidence and state your final diagnosis.
        3.3. **Key Indicators**: List the top 2-3 most critical indicators from the physical features that led to your final decision.
        </reasoning>

        ### **OUTPUT FORMAT**

        Provide your final answer in the JSON format below, enclosed within an `<answer>` block. Do not add any text or explanations outside the `<reasoning>` and `<answer>` blocks.

        <answer>{{
        "vib_only_label": "<The diagnosis from Step 1.2>",
        "vib_reason": "<A brief sentence explaining the token-based conclusion>",
        "knowledge_only_label": "<The diagnosis from Step 2.3>",
        "knowledge_reason": "<A brief sentence explaining the knowledge-based conclusion>",
        "criteria": "<The key indicators from Step 3.3>",
        "final_label": "<Your final diagnosis from Step 3.2>",
        "fusion_reason": "<A brief sentence explaining how you reconciled the two analyses to reach the final decision>"
        }}</answer>
        """

        prompt_only = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
        
        assistant_response = current_info['merged_class']
        if assistant_response == 'normal':
            assistant_response = 'normal(healthy)'
        elif assistant_response == 'bearing':
            assistant_response = 'bearing fault'

        return {
            'prompt': prompt_only,
            'answers': assistant_response,
            'current_x': current_x,
            'normal_x': normal_x
        }