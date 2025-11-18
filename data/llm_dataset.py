import os
import sys
from typing import Optional, Dict, List, Any
from torch.utils.data import Dataset
import numpy as np

# 프로젝트 루트를 sys.path에 추가 (data 디렉토리에서 실행할 때를 위해)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# tokenizers 병렬 처리 경고 방지 (프로세스 fork 전에 설정해야 함)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REBUILD_VECTORSTORE = False

from data.dataset import VibrationDataset
from legacy.GRPO_trainer.vllm_dataset import Planner, make_retriever
from feature_extract import LLM_Dataset as FeatureExtractLLMDataset


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
    
    def _summarize_text(text, max_words=100, max_chars=500):
        """텍스트 요약"""
        words = text.split()
        if len(words) > max_words:
            summary = " ".join(words[:50]) + " ... [중략] ... " + " ".join(words[-50:])
        else:
            summary = text
        
        if len(summary) > max_chars:
            summary = summary[:200] + " ... [중략] ... " + summary[-200:]
        return summary
    
    lines = []
    for idx, doc in enumerate(docs, 1):
        meta = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
        text = getattr(doc, "page_content", str(doc))
        
        section_info = _get_section_info(meta)
        summary = _summarize_text(text)
        
        lines.append(f"[DOC{idx}] {meta.get('source', 'unknown')} "
                     f"(청크 {meta.get('chunk_index', idx - 1)}, {section_info}): {summary}")
    return "\n".join(lines)


class LLM_Dataset(Dataset):
    """
    RAG 검색 결과와 프롬프트를 생성하는 데이터셋 클래스
    
    __getitem__에서 다음을 반환:
    - cur_status: 변화율 딕셔너리
    - rag_docs: Document 객체 리스트
    - rag_docs_formatted: 포맷된 문자열
    - user_prompt: VibrationSFTDataset 형식의 프롬프트 (Planner.plan() 내용 기반)
    """
    
    def __init__(self,
                 vibration_dataset: VibrationDataset,
                 retriever,
                 target_labels: str = "normal(healthy), misalignment, looseness, unbalance, bearing fault"):
        """
        Args:
            vibration_dataset: VibrationDataset 인스턴스
            retriever: 벡터 검색기 (Planner.retrieve()에서 사용)
            target_labels: 진단 대상 레이블 문자열
        """
        self.vibration_dataset = vibration_dataset
        self.retriever = retriever
        self.target_labels = target_labels
        
        self.feature_dataset = FeatureExtractLLMDataset(vibration_dataset=vibration_dataset)
        
        # Planner 인스턴스 생성 (retrieve만 사용하므로 더미 객체 사용)
        from unittest.mock import Mock
        self.planner = Planner(
            tokenizer=Mock(),
            llm=Mock(),
            retriever=retriever,
            max_tokens=4096,
            device="cpu"
        )
        
    def __len__(self):
        return len(self.vibration_dataset)

    def _create_prompt(self, current_knowledge: Dict[str, float], rag_docs_formatted: str) -> str:
        """Planner.plan()의 프롬프트 내용을 VibrationSFTDataset 형식으로 변환"""
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
                # 원본 VibrationDataset 데이터
                'x_vib': 현재 상태 진동 데이터,
                'x_stft': 현재 상태 STFT (있을 경우),
                'x_info': 현재 상태 정보 딕셔너리,
                'x_cls': 현재 상태 클래스 인덱스,
                'ref_vib': 참조 상태 진동 데이터 (있을 경우),
                'ref_stft': 참조 상태 STFT (있을 경우),
                'ref_info': 참조 상태 정보 딕셔너리 (있을 경우),
                'ref_cls': 참조 상태 클래스 인덱스 (있을 경우),
                
                # 특징 추출 결과
                'cur_status': 변화율 딕셔너리,
                'x_feat': 현재 상태 특징 딕셔너리,
                'ref_feat': 참조 상태 특징 딕셔너리 (있을 경우),
                
                # RAG 검색 결과
                'rag_docs': Document 객체 리스트,
                'rag_docs_formatted': 포맷된 문자열,
                
                # 프롬프트
                'user_prompt': User 프롬프트만,
                'prompt': 전체 프롬프트 (System + User + Assistant)
            }
        """
        # 원본 데이터 및 특징 추출
        data_dict = self.vibration_dataset[index]
        feature_sample = self.feature_dataset[index]
        
        # RAG 검색 및 프롬프트 생성
        cur_status = feature_sample.get('cur_status', {})
        rag_docs = self.planner.retrieve(cur_status)
        rag_docs_formatted = format_docs(rag_docs) if rag_docs else ""
        prompt = self._create_prompt(cur_status, rag_docs_formatted)
        
        # 결과 딕셔너리 구성
        return {
            **data_dict,  # 원본 VibrationDataset 데이터
            'cur_status': cur_status,
            'x_feat': feature_sample.get('x_feat', {}),
            'ref_feat': feature_sample.get('ref_feat', None),
            'rag_docs': rag_docs,
            'rag_docs_formatted': rag_docs_formatted,
            'user_prompt': prompt,
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
    print(f"\nrag_docs_formatted 길이: {len(sample['rag_docs_formatted'])}")
    print(f"\nuser_prompt 길이: {len(sample['user_prompt'])}")
    print(f"\nuser_prompt (처음 500자):\n{sample['user_prompt']}...")
    print("=" * 50)
