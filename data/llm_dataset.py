import os
import sys
import re
import shutil
from typing import Optional, Dict, List, Any
from torch.utils.data import Dataset

# 프로젝트 루트를 sys.path에 추가 (data 디렉토리에서 실행할 때를 위해)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# tokenizers 병렬 처리 경고 방지 (프로세스 fork 전에 설정해야 함)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REBUILD_VECTORSTORE = False

from data.dataset import VibrationDataset
from feature_extract import LLM_Dataset as FeatureExtractLLMDataset

# LangChain imports
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


def retrieve_documents(
    retriever,
    current_knowledge: Dict[str, float],
    target_labels: str = "normal(healthy), misalignment, looseness, unbalance, bearing fault"
) -> List[Document]:
    """
    큰 변화율과 비정상 지표에 집중한 검색 쿼리 생성 및 문서 검색
    
    Args:
        retriever: 벡터 검색기
        current_knowledge: 정상 상태 대비 변화율(%) 딕셔너리
            - 형식: {'rms_x': 3622.7905, 'kurtosis_x': -71.4348, ...}
            - 양수: 정상보다 증가 (예: 316.26% = 정상 대비 316% 증가)
            - 음수: 정상보다 감소 (예: -75.28% = 정상 대비 75% 감소)
            - 변화율 = (현재값 - 정상값) / 정상값 * 100
            - 0이 아닌 값만 포함됨 (feature_extract.py에서 필터링됨)
        target_labels: 진단 대상 레이블 문자열
    
    Returns:
        검색된 Document 객체 리스트
    """
    def classify_changes(change_rates, threshold_extreme=50.0, threshold_large=20.0, threshold_moderate=10.0):
        """변화율을 극단/큰/중간으로 분류하고 절댓값 기준 내림차순 정렬"""
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
    
    # 고장 유형별 중요 특징 매핑 테이블
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
    
    # 가중치 기반 고장 유형 추론
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
    
    # 검색 쿼리 생성 (비정상 특징 강조)
    query_parts = [
        "vibration diagnosis rotating machinery",
        f"diagnostic methods classify {target_labels}",
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
    
    # retriever의 k 값을 가져오기
    original_k = retriever.search_kwargs.get('k', 4) if hasattr(retriever, 'search_kwargs') else 4
    
    # k + 3개를 검색하도록 임시로 k 증가
    if hasattr(retriever, 'search_kwargs'):
        original_search_kwargs = retriever.search_kwargs.copy()
        retriever.search_kwargs['k'] = original_k + 3
    
    try:
        retrieved_docs = retriever.invoke(query)
    finally:
        # 원래 search_kwargs 복원
        if hasattr(retriever, 'search_kwargs'):
            retriever.search_kwargs = original_search_kwargs
    
    # 중복 제거: 소스 파일 + 청크 인덱스 조합으로 중복 제거
    seen_keys = set()
    unique_docs = []
    
    for doc in retrieved_docs:
        meta = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
        source = meta.get('source', 'unknown')
        chunk_idx = meta.get('chunk_index', -1)
        
        # 소스 파일과 청크 인덱스 조합으로 중복 판단
        key = (source, chunk_idx)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_docs.append(doc)
            # 원래 k개가 모이면 중단
            if len(unique_docs) >= original_k:
                break
    
    # 정확히 k개 반환 (부족하면 그대로 반환)
    return unique_docs[:original_k]


def make_retriever(
    embedding_model: str,
    model_cache: str,
    docs_path: str,
    retriever_k: int,
    rebuild: bool = False
):
    """
    벡터 스토어와 retriever 생성
    
    Args:
        embedding_model: 임베딩 모델명
        model_cache: 모델 캐시 경로
        docs_path: 문서 경로
        retriever_k: 검색할 문서 수
        rebuild: True면 기존 벡터 스토어 삭제 후 재생성, False면 기존 것 재사용
    
    Returns:
        Retriever 객체
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
        
        # docs_path 폴더에 있는 TXT 파일들을 불러오기
        txt_files = [os.path.join(docs_path, f) for f in os.listdir(docs_path) if f.lower().endswith('.txt')]
        if not txt_files:
            raise ValueError(f"docs_path에 .txt 파일이 없습니다: {docs_path}")
        
        raw_docs = []
        for path in txt_files:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                raw_docs.append(Document(
                    page_content=text,
                    metadata={'source': os.path.basename(path)}
                ))
        
        print(f"로드된 문서 수: {len(raw_docs)}")
        
        # SemanticTextSplitter로 문서 구조 기반 분할 (메타데이터 강화 포함)
        text_splitter = SemanticTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(raw_docs)
        print(f"생성된 청크 수: {len(docs)}")
        
        # VectorDB에 문서 저장
        print("벡터 스토어 생성 중...")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print(f"벡터 스토어 저장 완료: {persist_directory}")
    
    # MMR 검색을 사용하여 Retriever 생성 (검색 다양성 향상)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": retriever_k,
            "fetch_k": retriever_k * 3,  # 다양성 확보를 위해 더 많은 후보 검색
            "lambda_mult": 0.5  # 유사도와 다양성 균형 (0.0=다양성, 1.0=유사도)
        }
    )
    
    return retriever


def format_docs(docs):
    """검색된 문서를 포맷팅하여 표시 (청크 인덱스, 섹션 정보 포함)"""
    def _get_section_info(meta):
        """메타데이터에서 섹션 정보 추출"""
        parts = []
        if meta.get("chapter"): parts.append(f"Ch{meta['chapter']}")
        if meta.get("section"): parts.append(f"Sec{meta['section']}")
        if meta.get("subsection"): parts.append(f"Sub{meta['subsection']}")
        
        struct_num, struct_title = meta.get("struct_number"), meta.get("struct_title")
        content_type = meta.get("content_type", "text").capitalize()
        if struct_num and struct_title:
            parts.append(f"{content_type} {struct_num}: {struct_title}")
        elif struct_num:
            parts.append(f"{content_type} {struct_num}")
        
        return " | ".join(parts) if parts else "일반 텍스트"
    
    lines = []
    for idx, doc in enumerate(docs, 1):
        meta = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
        text = getattr(doc, "page_content", str(doc))
        
        section_info = _get_section_info(meta)
        
        lines.append(f"[DOC{idx}] {meta.get('source', 'unknown')} "
                     f"(청크 {meta.get('chunk_index', idx - 1)}, {section_info}):\n{text}")
    return "\n\n".join(lines)


class LLM_Dataset(Dataset):
    """
    RAG 검색 결과와 프롬프트를 생성하는 데이터셋 클래스
    
    __getitem__에서 다음을 반환:
    - cur_status: 변화율 딕셔너리
    - x_feat: 현재 상태 특징 딕셔너리
    - ref_feat: 참조 상태 특징 딕셔너리 (있을 경우)
    - rag_docs: Document 객체 리스트
    - prompt: 전체 프롬프트 (System + User + Assistant)
    
    """
    
    def __init__(self,
                 vibration_dataset: VibrationDataset,
                 retriever,
                 target_labels: str = "normal(healthy), misalignment, looseness, unbalance, bearing fault"):
        """
        Args:
            vibration_dataset: VibrationDataset 인스턴스
            retriever: 벡터 검색기
            target_labels: 진단 대상 레이블 문자열
        """
        self.vibration_dataset = vibration_dataset
        self.retriever = retriever
        self.target_labels = target_labels
        
        self.feature_dataset = FeatureExtractLLMDataset(vibration_dataset=vibration_dataset)
    
    def __len__(self):
        return len(self.vibration_dataset)

    def _create_prompt(self, current_knowledge: Dict[str, float], rag_docs_formatted: str) -> str:
        """프롬프트 생성"""
        # 변화율 딕셔너리를 문자열로 변환
        knowledge_str = "\n".join([f"{k}: {v:.4f}" for k, v in current_knowledge.items()]) \
            if isinstance(current_knowledge, dict) else str(current_knowledge)
        
        # 프롬프트 템플릿 구성
        system_prompt = (
            "You are a senior vibration analyst. Analyze the CURRENT STATE and provide SPECIFIC DIAGNOSIS "
            "based on actual change rate values. Be precise and cite sources. Answer in korean."
        )
        
        data_format_explanation = (
            "DATA FORMAT EXPLANATION:\n"
            "- Each line shows a feature name and its change rate percentage from normal baseline.\n"
            "- Format: \"feature_name: change_rate%\"\n"
            "- Positive values (e.g., 316.26%): The feature value is HIGHER than normal (increased by 316%)\n"
            "- Negative values (e.g., -75.28%): The feature value is LOWER than normal (decreased by 75%)\n"
            "- Change rate = (current_value - normal_value) / normal_value * 100\n"
            "- Example: \"order_x_2x: 316.26%\" means the 2nd harmonic component (order_x_2x) is 316% higher than the normal baseline.\n"
            "- Example: \"crest_factor_x: -75.28%\" means the crest factor is 75% lower than normal.\n"
        )
        
        reasoning_template = (
            "<think>\n"
            "MAIN STEP 1 — Embedding-based Comparison\n"
            "1.1 Compute or conceptually assess similarity/distance between <CURRENT_VIB_EMB> and <NORMAL_VIB_EMB> (e.g., cosine proximity).\n"
            "1.2 Identify which label prototypes the CURRENT embedding is closest to, if implied by features/notes.\n"
            "1.3 Summarize whether embeddings alone suggest normality or a specific fault class, and why.\n\n"
            "MAIN STEP 2 — SPECIFIC DIAGNOSIS CONCLUSION\n"
            "2.1 Based on the actual change rate values, determine the most likely fault type(s) with confidence level and specific reasoning.\n"
            "   - Use the actual percentage values in your reasoning (e.g., 'order_x_2x is 316% higher than normal')\n"
            "   - Cite specific values and compare them with diagnostic criteria from evidence snippets\n"
            "   - Be SPECIFIC about the current state using actual percentage values, not generic guidelines\n"
            "   - Cite sources from evidence snippets using [DOC#] format\n\n"
            "MAIN STEP 3 — DIAGNOSIS PLAN\n"
            "3.1 Provide general diagnostic guidelines for each fault type.\n"
            "   - For each fault type in the target labels, provide diagnosis rules\n"
            "   - Each rule should include: diagnosis idea, why (reason), and source (DOC#)\n"
            "   - Base the guidelines on the evidence snippets and current state analysis\n"
            "</think>\n"
        )
        
        answer_template = (
            "<answer>{\n"
            "  \"current_analysis\": {\n"
            "    \"abnormal_features\": [\n"
            "      {\"feature\": \"order_x_2x\", \"change_rate\": 316.26, \"category\": \"extreme\", \"interpretation\": \"2차 조화 성분이 정상 대비 316% 증가하여 축 불일치 징후\"}\n"
            "    ],\n"
            "    \"summary\": \"현재 상태 요약: 주요 비정상 특징과 의미\"\n"
            "  },\n"
            "  \"diagnosis_conclusion\": {\n"
            "    \"most_likely_fault\": \"misalignment\",\n"
            "    \"confidence\": \"high\",\n"
            "    \"reasoning\": \"order_x_2x가 정상 대비 316%, order_y_2x가 정상 대비 364% 증가하여 2차 조화 성분이 강하게 나타남. 이는 ISO 13373-1에 따른 축 불일치의 전형적 징후 [DOC4].\",\n"
            "    \"supporting_features\": [\"order_x_2x\", \"order_y_2x\"],\n"
            "    \"alternative_faults\": [{\"fault\": \"looseness\", \"reason\": \"crest_factor_x가 정상 대비 -69% 감소\"}]\n"
            "  },\n"
            f"  \"diagnosis_plan\": {{\n"
            f"    \"normal(healthy)\": [{{\"diagnosis idea\": \"<one line>\", \"why\": \"<short reason>\", \"source\": \"DOC#\"}}],\n"
            f"    \"misalignment\": [{{\"diagnosis idea\": \"<one line>\", \"why\": \"<short reason>\", \"source\": \"DOC#\"}}],\n"
            f"    \"looseness\": [{{\"diagnosis idea\": \"<one line>\", \"why\": \"<short reason>\", \"source\": \"DOC#\"}}],\n"
            f"    \"unbalance\": [{{\"diagnosis idea\": \"<one line>\", \"why\": \"<short reason>\", \"source\": \"DOC#\"}}],\n"
            f"    \"bearing fault\": [{{\"diagnosis idea\": \"<one line>\", \"why\": \"<short reason>\", \"source\": \"DOC#\"}}]\n"
            "  }\n"
            "}</answer>\n"
        )
        
        important_notes = (
            "IMPORTANT:\n"
            "- Use ACTUAL CHANGE RATE VALUES from CURRENT STATE in your analysis (e.g., 'order_x_2x is 316% higher than normal' not 'order_x_2x is high').\n"
            "- Always mention '정상 대비 X% 증가/감소' when referring to change rates.\n"
            "- Be SPECIFIC about the current state using actual percentage values, not generic guidelines.\n"
            "- Cite sources from evidence snippets using [DOC#] format.\n"
            "- All text must be in korean.\n"
            "Constraints: Only one think block and one answer block. No extra text."
        )
        
        user_prompt = (
            f"Diagnose the CURRENT STATE of rotating machinery among: {self.target_labels}.\n\n"
            f"CURRENT STATE ANALYSIS DATA:\n{knowledge_str}\n\n"
            f"{data_format_explanation}\n"
            f"Evidence snippets from manuals/papers (each prefixed with [DOC#]):\n{rag_docs_formatted}\n\n"
            f"TASK:\n"
            f"You will perform a 3-stage diagnostic reasoning process comparing NORMAL vs CURRENT states. "
            f"Follow the EXACT structure below. "
            f"Write detailed analysis in <think> and concise result in <answer>.\n\n"
            f"{reasoning_template}\n"
            f"{answer_template}\n"
            f"{important_notes}"
        )
        
        return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
    
    def __getitem__(self, index):
        """
        샘플을 가져와서 RAG 검색 및 프롬프트 생성
        
        Returns:
            dict: {
                # 특징 추출 결과
                'cur_status': 변화율 딕셔너리,
                'x_feat': 현재 상태 특징 딕셔너리,
                'ref_feat': 참조 상태 특징 딕셔너리 (있을 경우),
                
                # RAG 검색 결과
                'rag_docs': Document 객체 리스트,
                
                # 프롬프트
                'prompt': 전체 프롬프트 (System + User + Assistant)
            }
        """
        # 원본 데이터 특징 추출
        data_dict = self.vibration_dataset[index]
        feature_sample = self.feature_dataset[index]
        
        # RAG 검색 및 프롬프트 생성
        cur_status = feature_sample.get('cur_status', {})
        rag_docs = retrieve_documents(self.retriever, cur_status, self.target_labels)
        rag_docs_formatted = format_docs(rag_docs) if rag_docs else ""
        prompt = self._create_prompt(cur_status, rag_docs_formatted)
        
        # 결과 딕셔너리 구성
        return {
            'cur_status': cur_status,
            'x_feat': feature_sample.get('x_feat', {}),
            'ref_feat': feature_sample.get('ref_feat', None),
            'rag_docs': rag_docs,
            'prompt': prompt
        }


def get_llm_dataset(train_dataset, val_dataset, retriever, 
                    target_labels: str = "normal(healthy), misalignment, looseness, unbalance, bearing fault"):
    """Train/Val LLM_Dataset 생성 편의 함수"""
    create_dataset = lambda ds: LLM_Dataset(vibration_dataset=ds, retriever=retriever, target_labels=target_labels)
    return create_dataset(train_dataset), create_dataset(val_dataset)


def create_retrieve_dataset(
    data_root: str,
    docs_path: str = "docs_path",
    window_sec: float = 5.0,
    stride_sec: float = 2.5,
    using_dataset: list = ['dxai', 'iis', 'vat', 'vbl', 'mfd'],
    include_ref: bool = True,
    transform=None,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_cache: Optional[str] = None,
    retriever_k: int = 4,
    rebuild_vectorstore: bool = False,
    target_labels: str = "normal(healthy), misalignment, looseness, unbalance, bearing fault",
    drop_last: bool = True,
    dtype=None,
    channel_order: tuple = ("x", "y"),
    test_mode: bool = False
) -> LLM_Dataset:
    """
    LLM_Dataset을 생성하는 편의 함수 (VibrationDataset도 내부에서 생성)
    
    Args:
        data_root: 데이터셋 루트 경로
        docs_path: 문서 디렉토리 경로
        window_sec: 윈도우 길이 (초)
        stride_sec: 스트라이드 길이 (초)
        using_dataset: 사용할 데이터셋 리스트
        include_ref: 참조 데이터 포함 여부
        transform: 변환 함수 (STFT 등)
        embedding_model: 임베딩 모델명
        model_cache: 모델 캐시 경로 (None이면 ~/.cache/huggingface 사용)
        retriever_k: 검색할 문서 수
        rebuild_vectorstore: 벡터 DB 재생성 여부
        target_labels: 진단 대상 레이블 문자열
        drop_last: 마지막 배치 버리기 여부
        dtype: 데이터 타입
        channel_order: 채널 순서
        test_mode: 테스트 모드 여부
        
    Returns:
        LLM_Dataset 인스턴스
    """
    model_cache = model_cache or os.path.expanduser("~/.cache/huggingface")
    os.makedirs(model_cache, exist_ok=True)
    
    # VibrationDataset 및 Retriever 생성
    vibration_dataset = VibrationDataset(
        data_root=data_root, window_sec=window_sec, stride_sec=stride_sec,
        using_dataset=using_dataset, drop_last=drop_last, dtype=dtype,
        transform=transform, channel_order=channel_order, test_mode=test_mode,
        include_ref=include_ref
    )
    
    retriever = make_retriever(
        embedding_model=embedding_model, model_cache=model_cache,
        docs_path=docs_path, retriever_k=retriever_k, rebuild=rebuild_vectorstore
    )
    
    return LLM_Dataset(vibration_dataset=vibration_dataset, retriever=retriever, target_labels=target_labels)


if __name__ == "__main__":
    # 테스트 코드
    # 프로젝트 루트 기준으로 경로 설정 (data 디렉토리에서 실행할 때를 위해)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 데이터셋 경로 확인
    dataset_path = os.path.join(project_root, "data", "dataset")
    meta_csv_path = os.path.join(dataset_path, "meta.csv")
    if not os.path.exists(meta_csv_path):
        print(f"⚠️  경고: {meta_csv_path} 파일을 찾을 수 없습니다.")
        exit(1)
    
    # LLM_Dataset 생성 (VibrationDataset도 내부에서 생성)
    dataset_path = os.path.join(project_root, "data", "dataset")
    docs_path = os.path.join(project_root, "docs_path")
    retrieve_dataset = create_retrieve_dataset(
        data_root=dataset_path,
        docs_path=docs_path,
        window_sec=5.0,
        stride_sec=2.5,
        using_dataset=['dxai'],
        include_ref=True,
        transform=None,
        retriever_k=4,
        rebuild_vectorstore=REBUILD_VECTORSTORE
    )
    
    # 첫 번째 샘플 테스트
    print("=" * 50)
    print("LLM_Dataset 테스트")
    print("=" * 50)
    
    sample = retrieve_dataset[0]
    
    print(f"\ncur_status (처음 5개): {list(sample['cur_status'].items())[:5]}")
    print(f"\nrag_docs 개수: {len(sample['rag_docs'])}")
    print(f"\nprompt 길이: {len(sample['prompt'])}")
    print(f"\nprompt (처음 500자):\n{sample['prompt'][:500]}...")
    print("=" * 50)
