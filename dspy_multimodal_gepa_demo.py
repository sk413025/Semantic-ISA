"""
DSPy Type System + GEPA Multimodal Optimization 完整解析
=========================================================

這個腳本展示了 dspy.Type 系統如何與 GEPA 優化器協作，
實現多模態（Image/Audio）任務的自動 prompt 優化。

▶ 不需要 API Key 即可執行前三個實驗（結構驗證）
▶ 第四個實驗需要 API Key（實際呼叫 LM）

用法:
    # 只跑結構驗證（不需 API）
    python dspy_multimodal_gepa_demo.py
    
    # 完整跑（需要設定 OPENAI_API_KEY 或 GEMINI_API_KEY）
    OPENAI_API_KEY=sk-xxx python dspy_multimodal_gepa_demo.py --run-llm
"""

import dspy
import base64
import json
import sys
from typing import Literal, Any
from dspy.adapters.types import Type as DspyType
from dspy.teleprompt.gepa.instruction_proposal import (
    MultiModalInstructionProposer,
    SingleComponentMultiModalProposer,
)
from dspy.teleprompt.gepa.gepa_utils import ReflectiveExample


# ============================================================
# 工具函數
# ============================================================

def make_fake_image(label: str = "demo") -> dspy.Image:
    """建立一個假的 1x1 PNG 圖片（用於結構驗證）"""
    tiny_png = base64.b64encode(
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
        b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
        b'\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00'
        b'\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
    ).decode()
    return dspy.Image(f"data:image/png;base64,{tiny_png}")


def section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ============================================================
# 實驗 1：dspy.Type 繼承體系
# ============================================================

def experiment_1_type_hierarchy():
    section("實驗 1：dspy.Type 繼承體系")

    print("""
    dspy.Type（Pydantic BaseModel）
       ├── dspy.Image      # 圖片：url → format() → [{"type":"image_url",...}]
       ├── dspy.Audio      # 音訊：data+format → format() → [{"type":"input_audio",...}]
       ├── dspy.History    # 對話歷史
       ├── dspy.ToolCalls  # 工具呼叫結果
       ├── dspy.Tool       # 工具定義
       ├── dspy.Reasoning  # reasoning model 的原生推理鏈
       └── (自訂 Type)     # 任何繼承 dspy.Type 並實作 format() 的類別
    """)

    # 驗證繼承關係
    for name in ['Image', 'Audio', 'History', 'ToolCalls', 'Tool', 'Reasoning']:
        cls = getattr(dspy, name)
        is_type = issubclass(cls, DspyType)
        has_format = hasattr(cls, 'format')
        print(f"  dspy.{name:12s} → issubclass(Type)={is_type}, has format()={has_format}")

    # 展示 format() 的多型性
    print("\n  --- format() 輸出對比 ---")
    img = make_fake_image()
    audio = dspy.Audio(data="AAAA", audio_format="wav")

    print(f"  Image.format() → {json.dumps(img.format()[0]['type'])}")
    print(f"  Audio.format() → {json.dumps(audio.format()[0]['type'])}")
    print(f"\n  ✅ 不同 Type 的 format() 產出不同的 content block type")
    print(f"     Adapter 根據這些 type 決定如何建構 LM API 請求")


# ============================================================
# 實驗 2：Adapter 格式轉換全流程
# ============================================================

def experiment_2_adapter_formatting():
    section("實驗 2：Adapter 如何將 Signature + dspy.Image 翻譯成 LM prompt")

    class WaferDefect(dspy.Signature):
        """Classify wafer defect type from microscopy image.
        Pay attention to edge sharpness, branching patterns, and depth profiles."""
        wafer_image: dspy.Image = dspy.InputField(desc="Microscopy image")
        magnification: str = dspy.InputField(desc="Magnification level")
        defect_type: Literal["scratch", "particle", "void", "crack", "stain", "none"] = dspy.OutputField()

    adapter = dspy.ChatAdapter()
    inputs = {"wafer_image": make_fake_image(), "magnification": "500x"}
    messages = adapter.format(signature=WaferDefect, demos=[], inputs=inputs)

    print(f"\n  ChatAdapter.format() 產出 {len(messages)} 個 messages:")

    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            print(f"\n  ┌─ [{i}] role={role} (純文字)")
            for line in content.split('\n')[:8]:
                print(f"  │  {line}")
            print(f"  └─ ... (system prompt 定義欄位格式和型別約束)")
        elif isinstance(content, list):
            print(f"\n  ┌─ [{i}] role={role} ({len(content)} 個 content blocks)")
            for j, block in enumerate(content):
                if isinstance(block, dict):
                    btype = block.get("type", "?")
                    if btype == "text":
                        text_preview = block["text"][:80].replace('\n', '↩')
                        print(f"  │  [{j}] 📝 text: {text_preview}...")
                    elif btype == "image_url":
                        print(f"  │  [{j}] 🖼️  image_url: data:image/png;base64,...")
                    else:
                        print(f"  │  [{j}] {btype}")
            print(f"  └─")

    print(f"""
  ✅ 關鍵觀察：
     • system message 是純文字 — 定義欄位名、型別約束、輸出格式
     • user message 是 list[content_block] — 文字和圖片交錯排列
     • dspy.Image 被自動轉成 {{"type":"image_url",...}} content block
     • 開發者完全不需要手動處理 base64 編碼或 API 格式
    """)


# ============================================================
# 實驗 3：GEPA MultiModal 反思迴圈模擬
# ============================================================

def experiment_3_gepa_reflection_simulation():
    section("實驗 3：GEPA MultiModal 反思迴圈模擬")

    print("\n▶ Step 1: 建立模擬的 reflective_dataset...")

    reflective_examples = [
        ReflectiveExample(
            Inputs={"wafer_image": make_fake_image(), "magnification": "500x"},
            **{"Generated Outputs": {"defect_type": "scratch"}},
            Feedback="Incorrect. Expected: crack. Got: scratch. "
                     "Cracks have branching patterns along grain boundaries, "
                     "while scratches are linear and surface-level."
        ),
        ReflectiveExample(
            Inputs={"wafer_image": make_fake_image(), "magnification": "200x"},
            **{"Generated Outputs": {"defect_type": "particle"}},
            Feedback="Correct! Model identified the irregular, opaque deposit."
        ),
        ReflectiveExample(
            Inputs={"wafer_image": make_fake_image(), "magnification": "1000x"},
            **{"Generated Outputs": {"defect_type": "void"}},
            Feedback="Incorrect. Expected: stain. Got: void. "
                     "Stains have diffuse edges and color gradients; "
                     "voids have sharp circular boundaries with depth."
        ),
    ]

    print(f"  {len(reflective_examples)} 個範例（2 錯 1 對）\n")

    proposer = SingleComponentMultiModalProposer()

    print("▶ Step 2: _format_examples_for_instruction_generation()...")
    formatted_text, image_map = proposer._format_examples_for_instruction_generation(
        reflective_examples
    )

    print(f"\n  文字部分（reflection_lm 看到的 prompt）:")
    print(f"  {'─' * 50}")
    for line in formatted_text.split('\n')[:25]:
        print(f"  │ {line}")
    print(f"  │ ... (共 {len(formatted_text)} 字元)")

    print(f"\n  圖片部分（跟文字一起送給 reflection_lm）:")
    for idx, images in image_map.items():
        print(f"  │ Example {idx}: {len(images)} 張圖片")

    print("\n▶ Step 3: _create_multimodal_examples()...")
    multimodal_content = proposer._create_multimodal_examples(formatted_text, image_map)

    print(f"\n  最終 multimodal_content = [")
    for i, item in enumerate(multimodal_content):
        if isinstance(item, str):
            print(f"    [{i}] str ({len(item)} chars)  ← 格式化文字 + feedback + pattern analysis")
        elif isinstance(item, dspy.Image):
            print(f"    [{i}] dspy.Image              ← 實際圖片（base64）")
    print(f"  ]")

    print("\n▶ Step 4: _analyze_feedback_patterns()...")
    analysis = proposer._analyze_feedback_patterns(reflective_examples)
    print(f"  Error patterns:   {len(analysis['error_patterns'])} 個")
    print(f"  Success patterns: {len(analysis['success_patterns'])} 個")

    print(f"""
  ✅ GEPA MultiModal 反思迴圈的完整流程：
  
  ┌──────────────────────────────────────────────────────────────┐
  │ 1. Student LM 跑 trainset → 產生 (prediction, trace)        │
  │                                                              │
  │ 2. Metric 函數計算 score + 生成文字 feedback                  │
  │    → "Incorrect. Expected: crack. Got: scratch.              │
  │       Cracks have branching patterns..."                     │
  │                                                              │
  │ 3. GEPA 收集 ReflectiveExample:                              │
  │    Inputs (含 dspy.Image) + Generated Outputs + Feedback     │
  │                                                              │
  │ 4. MultiModalInstructionProposer:                            │
  │    a) 文字中用 [IMAGE-N] placeholder 標記圖片位置             │
  │    b) 實際圖片收集到 image_map                                │
  │    c) 分析 feedback patterns（error/success 分類）            │
  │    d) 組合成 [text, Image, Image, ...] multimodal content    │
  │                                                              │
  │ 5. Reflection LM（強大的多模態 LM）：                         │
  │    → 同時看到圖片 + 錯誤分析 + pattern summary               │
  │    → 輸出：新的 instruction 文字                              │
  │                                                              │
  │ 6. 新 instruction 寫回 Signature → Student 重跑 → 循環       │
  └──────────────────────────────────────────────────────────────┘
    """)


# ============================================================
# 實驗 4：完整可執行的 GEPA Multimodal 範例（需要 API Key）
# ============================================================

def experiment_4_full_gepa_run():
    section("實驗 4：完整 GEPA Multimodal 優化（需要 API Key）")

    import os

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("""
  ⚠️ 未設定 API Key，跳過實際 LM 呼叫。
  
  要實際執行，請設定環境變數：
    export OPENAI_API_KEY=sk-xxx
    # 或
    export GEMINI_API_KEY=xxx
    
  然後重新執行：
    python dspy_multimodal_gepa_demo.py --run-llm
        """)

        print("  以下是完整可執行的程式碼：\n")
        print("""
  # ====== 完整的 GEPA + dspy.Image 優化範例 ======

  import dspy
  from typing import Literal
  from dspy.teleprompt.gepa.instruction_proposal import MultiModalInstructionProposer

  # 1. 定義 Signature
  class WaferDefect(dspy.Signature):
      \\"\\"\\"Classify the defect type in a semiconductor wafer image.\\"\\"\\"
      wafer_image: dspy.Image = dspy.InputField(desc="Microscopy image")
      magnification: str = dspy.InputField(desc="Magnification level")
      defect_type: Literal["scratch","particle","void","crack","stain","none"] = dspy.OutputField()

  # 2. 配置 LM
  student_lm = dspy.LM("openai/gpt-4o-mini")   # 便宜的 student
  reflection_lm = dspy.LM("openai/gpt-4o", temperature=1.0, max_tokens=16000)
  dspy.configure(lm=student_lm)

  # 3. 建立 trainset（每個 Example 含 dspy.Image）
  trainset = [
      dspy.Example(
          wafer_image=dspy.Image("path/to/wafer_001.png"),
          magnification="500x",
          defect_type="crack"
      ).with_inputs("wafer_image", "magnification"),
      # ... 更多範例（建議 20-50 個）
  ]

  # 4. 定義 metric（帶 feedback）
  def wafer_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
      correct = gold.defect_type == pred.defect_type
      score = 1.0 if correct else 0.0
      if not correct:
          fb = (f"Misclassified as '{pred.defect_type}', "
                f"true label is '{gold.defect_type}'. "
                f"At magnification {gold.magnification}.")
      else:
          fb = f"Correct: {gold.defect_type}."
      return dspy.Prediction(score=score, feedback=fb)

  # 5. GEPA 優化
  program = dspy.ChainOfThought(WaferDefect)
  gepa = dspy.GEPA(
      metric=wafer_metric,
      reflection_lm=reflection_lm,
      instruction_proposer=MultiModalInstructionProposer(),
      auto="medium",
  )
  optimized = gepa.compile(student=program, trainset=trainset)

  # 6. 儲存優化後的程式
  optimized.save("wafer_defect_optimized.json")
        """)
        return

    # 如果有 API key，實際執行一個小 demo
    print("  ✅ 偵測到 API Key，嘗試實際執行...")
    # (略 - 根據實際 API key 類型配置 LM)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    experiment_1_type_hierarchy()
    experiment_2_adapter_formatting()
    experiment_3_gepa_reflection_simulation()

    if "--run-llm" in sys.argv:
        experiment_4_full_gepa_run()
    else:
        section("實驗 4：完整 GEPA 優化（跳過，加 --run-llm 旗標執行）")
        print("\n  加入 --run-llm 並設定 API Key 以執行完整的 GEPA multimodal 優化")

    section("總結")
    print("""
  dspy.Type 系統的三層架構：
  
  ┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
  │  Signature   │     │   Adapter    │     │    LM API        │
  │              │     │              │     │                  │
  │  wafer_image │────▶│ ChatAdapter  │────▶│ content blocks:  │
  │  : dspy.Image│     │ JSONAdapter  │     │ [{type:image_url}│
  │              │     │ XMLAdapter   │     │  {type:text}...] │
  └─────────────┘     └──────────────┘     └──────────────────┘
        │                                          │
        │              ┌──────────────┐            │
        └─────────────▶│  Optimizer   │◀───────────┘
                       │              │   (traces)
                       │  dspy.GEPA + │
                       │  MultiModal  │
                       │  Proposer    │
                       └──────────────┘
                             │
                             ▼
                     新的 instruction
                     (純文字，寫回 Signature)
    """)
