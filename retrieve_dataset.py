import os
from typing import Optional, Dict, List, Any
from torch.utils.data import Dataset

# tokenizers 병렬 처리 경고 방지 (프로세스 fork 전에 설정해야 함)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REBUILD_VECTORSTORE = False

from legacy.GRPO_trainer.vllm_dataset import Planner, make_retriever
from feature_extract import LLM_Dataset
from data.dataset import VibrationDataset


def format_docs(docs):
    """검색된 문서를 포맷팅하여 표시 (청크 인덱스, 섹션 정보 포함)"""
    lines = []
    for idx, doc in enumerate(docs, 1):
        meta = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
        source = meta.get("source", "unknown")
        chunk_index = meta.get("chunk_index", idx - 1)
        chapter = meta.get("chapter", None)
        section = meta.get("section", None)
        subsection = meta.get("subsection", None)
        content_type = meta.get("content_type", "text")
        struct_number = meta.get("struct_number", None)
        struct_title = meta.get("struct_title", None)
        
        text = getattr(doc, "page_content", str(doc))
        
        # 섹션 정보 구성
        section_info_parts = []
        if chapter:
            section_info_parts.append(f"Ch{chapter}")
        if section:
            section_info_parts.append(f"Sec{section}")
        if subsection:
            section_info_parts.append(f"Sub{subsection}")
        if struct_number and struct_title:
            section_info_parts.append(f"{content_type.capitalize()} {struct_number}: {struct_title}")
        elif struct_number:
            section_info_parts.append(f"{content_type.capitalize()} {struct_number}")
        
        section_info = " | ".join(section_info_parts) if section_info_parts else "일반 텍스트"
        
        # 텍스트 요약 (처음과 끝 부분 표시)
        text_words = text.split()
        if len(text_words) > 100:
            # 처음 50단어 + 중략 + 끝 50단어
            summary = " ".join(text_words[:50]) + " ... [중략] ... " + " ".join(text_words[-50:])
        else:
            summary = text
        
        # 최대 길이 제한
        if len(summary) > 500:
            summary = summary[:200] + " ... [중략] ... " + summary[-200:]
        
        lines.append(f"[DOC{idx}] {source} (청크 {chunk_index}, {section_info}): {summary}")
    return "\n".join(lines)


class RetrieveDataset(Dataset):
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
        
        # 특징 추출을 위한 LLM_Dataset
        self.feature_dataset = LLM_Dataset(vibration_dataset=vibration_dataset)
        
        # Planner 인스턴스 생성 (retrieve만 사용)
        # retrieve 메서드만 사용하므로 tokenizer, llm은 더미 객체로 설정
        # Planner.retrieve()는 self.retriever만 사용하므로 None 전달 가능
        from unittest.mock import Mock
        dummy_tokenizer = Mock()
        dummy_llm = Mock()
        
        self.planner = Planner(
            tokenizer=dummy_tokenizer,
            llm=dummy_llm,
            retriever=retriever,
            max_tokens=4096,
            device="cpu"
        )
    
    def __len__(self):
        return len(self.vibration_dataset)
    
    def _create_prompt(self, current_knowledge: Dict[str, float], rag_docs_formatted: str) -> str:
        """
        Planner.plan()의 프롬프트 내용을 VibrationSFTDataset 형식으로 변환
        
        Args:
            current_knowledge: 변화율 딕셔너리
            rag_docs_formatted: 포맷된 RAG 검색 결과 문자열
            
        Returns:
            user_prompt: 변환된 프롬프트 문자열
        """
        # current_knowledge를 문자열로 변환
        if isinstance(current_knowledge, dict):
            knowledge_str = "\n".join([f"{k}: {v:.4f}" for k, v in current_knowledge.items()])
        else:
            knowledge_str = str(current_knowledge)
        
        # System 프롬프트
        system_prompt = (
            "You are a senior vibration analyst. Analyze the CURRENT STATE and provide SPECIFIC DIAGNOSIS "
            "based on actual change rate values. Be precise and cite sources. Answer in korean."
        )
        
        # User 프롬프트 (Planner.plan() 내용을 VibrationSFTDataset 형식으로 변환)
        user_prompt = (
            f"Diagnose the CURRENT STATE of rotating machinery among: {self.target_labels}.\n\n"
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
            f"{rag_docs_formatted}\n\n"
            "TASK:\n"
            "You will perform a 3-stage diagnostic reasoning process comparing NORMAL vs CURRENT states. "
            "Follow the EXACT structure below. "
            "Write detailed analysis in <think> and concise result in <answer>.\n\n"
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
            "</think>\n\n"
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
            "}</answer>\n\n"
            "IMPORTANT:\n"
            "- Use ACTUAL CHANGE RATE VALUES from CURRENT STATE in your analysis (e.g., 'order_x_2x is 316% higher than normal' not 'order_x_2x is high').\n"
            "- Always mention '정상 대비 X% 증가/감소' when referring to change rates.\n"
            "- Be SPECIFIC about the current state using actual percentage values, not generic guidelines.\n"
            "- Cite sources from evidence snippets using [DOC#] format.\n"
            "- All text must be in korean.\n"
            "Constraints: Only one think block and one answer block. No extra text."
        )
        
        prompt_only = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
        
        return prompt_only
    
    def __getitem__(self, idx):
        """
        샘플을 가져와서 RAG 검색 및 프롬프트 생성
        
        Returns:
            dict: {
                'cur_status': 변화율 딕셔너리,
                'rag_docs': Document 객체 리스트,
                'rag_docs_formatted': 포맷된 문자열,
                'user_prompt': 변환된 프롬프트
            }
        """
        # 1. 특징 추출
        feature_sample = self.feature_dataset[idx]
        cur_status = feature_sample.get('cur_status', {})  # 변화율 딕셔너리
        
        # 2. RAG 검색 수행
        rag_docs = self.planner.retrieve(cur_status)
        
        # 3. 문서 포맷팅
        rag_docs_formatted = format_docs(rag_docs) if rag_docs else ""
        
        # 4. 프롬프트 생성
        user_prompt = self._create_prompt(cur_status, rag_docs_formatted)
        
        return {
            'cur_status': cur_status,
            'rag_docs': rag_docs,
            'rag_docs_formatted': rag_docs_formatted,
            'user_prompt': user_prompt
        }


def create_retrieve_dataset(
    vibration_dataset: VibrationDataset,
    docs_path: str = "docs_path",
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_cache: Optional[str] = None,
    retriever_k: int = 4,
    rebuild_vectorstore: bool = False,
    target_labels: str = "normal(healthy), misalignment, looseness, unbalance, bearing fault"
) -> RetrieveDataset:
    """
    RetrieveDataset을 생성하는 편의 함수
    
    Args:
        vibration_dataset: VibrationDataset 인스턴스
        docs_path: 문서 디렉토리 경로
        embedding_model: 임베딩 모델명
        model_cache: 모델 캐시 경로 (None이면 ~/.cache/huggingface 사용)
        retriever_k: 검색할 문서 수
        rebuild_vectorstore: 벡터 DB 재생성 여부
        target_labels: 진단 대상 레이블 문자열
        
    Returns:
        RetrieveDataset 인스턴스
    """
    if model_cache is None:
        model_cache = os.path.expanduser("~/.cache/huggingface")
    
    os.makedirs(model_cache, exist_ok=True)
    
    # Retriever 생성
    retriever = make_retriever(
        embedding_model=embedding_model,
        model_cache=model_cache,
        docs_path=docs_path,
        retriever_k=retriever_k,
        rebuild=rebuild_vectorstore
    )
    
    # RetrieveDataset 생성
    dataset = RetrieveDataset(
        vibration_dataset=vibration_dataset,
        retriever=retriever,
        target_labels=target_labels
    )
    
    return dataset


if __name__ == "__main__":
    # 테스트 코드
    import os
    
    # 데이터셋 경로 확인
    dataset_path = "data/dataset"
    meta_csv_path = os.path.join(dataset_path, "meta.csv")
    if not os.path.exists(meta_csv_path):
        print(f"⚠️  경고: {meta_csv_path} 파일을 찾을 수 없습니다.")
        exit(1)
    
    # VibrationDataset 생성
    vibration_dataset = VibrationDataset(
        data_root=dataset_path,
        window_sec=5.0,
        stride_sec=2.5,
        using_dataset=['dxai'],
        include_ref=True,
        transform=None
    )
    
    # RetrieveDataset 생성
    retrieve_dataset = create_retrieve_dataset(
        vibration_dataset=vibration_dataset,
        docs_path="docs_path",
        retriever_k=4,
        rebuild_vectorstore=REBUILD_VECTORSTORE
    )
    
    # 첫 번째 샘플 테스트
    print("=" * 50)
    print("RetrieveDataset 테스트")
    print("=" * 50)
    
    sample = retrieve_dataset[0]
    
    print(f"\ncur_status (처음 5개): {list(sample['cur_status'].items())[:5]}")
    print(f"\nrag_docs 개수: {len(sample['rag_docs'])}")
    print(f"\nrag_docs_formatted 길이: {len(sample['rag_docs_formatted'])}")
    print(f"\nuser_prompt 길이: {len(sample['user_prompt'])}")
    print(f"\nuser_prompt (처음 500자):\n{sample['user_prompt']}...")
    print("=" * 50)

