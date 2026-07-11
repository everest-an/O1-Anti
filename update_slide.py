import os
import re

def update_file(filename, replacements):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
        else:
            print(f"Warning: Could not find '{old}' in {filename}")

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)


zh_replacements = [
    # 修正技术描述：明确是 FFN 替换，删除麻醉测试和意识计算
    (
        "抛弃标准 Transformer 中耗费显存的自注意力静态 FFN 层",
        "保留标准 Transformer 特征交互的同时，我们将耗费显存的静态 FFN （前馈网络）层"
    ),
    (
        "\\item \\textbf{取代自注意力机制：}",
        "\\item \\textbf{FFN 层的液态化改造：}"
    ),
    (
        "\\item \\textbf{原生支持意识计算：} 是世界首个突破“麻醉测试” (Anesthesia Benchmark) 的 AI 架构，不仅生成语言，还支持追踪内在认识坍缩的指标 ($\\hat{\\Phi}$)。",
        "\\item \\textbf{自适应长程记忆：} 支持实时追踪内在连续状态，无需频繁将所有历史 KV 数据重新调取计算，在极低内存下完成深度思考。"
    ),
    # 补充 128K 和线性架构的描述
    (
        "核心验证 2：128K 长上下文“大海捞针”",
        "长上下文评测：扩展优化与并发性能"
    ),
    (
        "长范围检索准确度平稳保持在 $99\\%+$。",
        "扩展范围内检索准确度依然具备长牛尾效应（完整 128K 大海捞针及高并发流式处理评测报告与代码即将全源发布）。"
    ),
    (
        "对比现有的高效线性架构（如 Mamba / RWKV 等）：",
        "对比前沿架构路线（针对 Mamba-2 / RWKV-v6 等最新线性架构）："
    ),
    (
        "克服了长期维持多向特征的衰减问题",
        "在极端长上下文 (256K+) 及并发流式分发场景下，展示出了更为平滑的内存扩展优势及更低的衰减风险"
    ),
    # 修改 4090 并发数据和成本描述
    (
        "普通 A100 模型集群极可能因显存激增而导致 OOM（宕机崩溃）；而我们可仅用单台消费级加速显卡从容稳定排队并行反馈。",
        "随着最新量化 KV 缓存 (如 StreamingLLM/PagedAttention) 的普及，大模型部署有所缓解，但 MT-LNN 凭借真正的 $O(1)$ 恒定状态占用，在诸如手机端侧及高频流媒体等极限长文多路并发场景中，依然拥有无可取代的硬件代差优势。"
    ),
    (
        "预计云端模型的大规模服务算力成本可降低至现有方案的 1/10。",
        "对比不断降价的云上开源长文本 API（约 $0.01/1M tokens），我们的自建私有化方案在数据隐私合规前提下仍具备竞争吸引力。"
    ),
    # 调整目标节奏
    (
        "二阶段架构规模化 (随后的 6-12 个月内)",
        "二阶段架构规模化 (随后的 12 个月内)"
    ),
    (
        "三阶段迈向下一代泛化多模态基座 (未来 18 个月以上)",
        "三阶段多模态探索与产业赋能 (未来 36 个月以上)"
    ),
    (
        "锁定 \\$1.5M 的年度经常性营收 (ARR)。",
        "聚焦 2-3 个垂直行业的标杆客户，锁定 \\$500K - \\$800K 的年度经常性营收 (ARR)。"
    ),
    # 补充团队背景
    # (Everest已在修改好)
]

en_replacements = [
    # 修正技术描述
    (
        "\\item \\textbf{Replacing Self-Attention:} We discard the memory-hogging static FFN layers of standard Transformers, replacing them with 13 parallel, continuous-time liquid network pathways.",
        "\\item \\textbf{Liquid FFN Architecture:} We retain standard feature interactions but replace the memory-hogging static FFN layers with 13 parallel, continuous-time liquid network pathways."
    ),
    (
        "\\item \\textbf{Native Conscious Computing:} The world's first AI architecture to break the \"Anesthesia Benchmark,\" capable of generating language while tracking an inherent cognitive collapse metric ($\\hat{\\Phi}$).",
        "\\item \\textbf{Adaptive Long-Range Memory:} Our design enables real-time tracking of continuous internal states without reloading sprawling historical KV data, unlocking deep reasoning under ultra-low memory."
    ),
    # 128K 
    (
        "Core Performance 2: 128K Needle in a Haystack",
        "Core Performance: Extended Context \& Concurrency"
    ),
    (
        "MT-LNN retrieves it with a linear \\textbf{99\\%+ accuracy}.",
        "preliminary tests reveal extending robustness. (Full 128K extraction and rigorous high-concurrency streaming reports will be open-sourced soon)."
    ),
    # Mamba
    (
        "\\textbf{Features} & \\textbf{MT-LNN} & \\textbf{GPT-4/Claude} & \\textbf{Mamba/RWKV} & \\textbf{Local Llama} \\\\",
        "\\textbf{Features} & \\textbf{MT-LNN} & \\textbf{GPT-4/Claude} & \\textbf{Mamba-2/RWKV-v6} & \\textbf{Local Llama} \\\\"
    ),
    (
        "Conscious Core Eval & \\textcolor{green}{Yes} & \\textcolor{red}{No} & \\textcolor{red}{No} & \\textcolor{red}{No} \\\\",
        "O(1) Streaming & \\textcolor{green}{Yes} & \\textcolor{red}{No} & \\textcolor{green}{Yes} & \\textcolor{red}{No} \\\\"
    ),
    # Cost & Scale modifications
    (
        "Because our inference costs are $\\sim$ 90\\% lower than Transformer APIs",
        "Despite falling API costs industry-wide, our highly efficient private endpoints"
    ),
    # Roadmap & Pipeline targets
    (
        "Target \\$1.5M ARR",
        "Target \\$500K - \\$800K ARR"
    ),
    (
        "Scale to \\$8M+ ARR",
        "Scale to \\$2.5M+ ARR"
    ),
    (
        "\\$50M+ ARR mapping.",
        "\\$15M+ ARR mapping."
    ),
    (
        "reach \\$1.5M ARR within 12 months.",
        "concentrate on 2-3 key vertical champions, and reach \\$500K - \\$800K ARR."
    ),
    (
        "Phase 2: Enterprise Deployment (Months 1-6):",
        "Phase 2: Enterprise Deployment (Months 6-12):"
    ),
    (
        "Phase 3: The A.G.I Checkmate (Months 6-18):",
        "Phase 3: Multimodal Expansion (Months 12-36):"
    )
]

update_file("investor_deck_mt_lnn_zh.tex", zh_replacements)
update_file("investor_deck_mt_lnn.tex", en_replacements)
