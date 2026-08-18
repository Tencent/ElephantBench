const header = document.querySelector("[data-header]");
const navToggle = document.querySelector(".nav-toggle");
const nav = document.querySelector(".site-nav");

const updateHeader = () => {
  header?.classList.toggle("scrolled", window.scrollY > 24);
};

updateHeader();
window.addEventListener("scroll", updateHeader, { passive: true });

navToggle?.addEventListener("click", () => {
  const isOpen = nav?.classList.toggle("open") ?? false;
  navToggle.setAttribute("aria-expanded", String(isOpen));
});

nav?.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => {
    nav.classList.remove("open");
    navToggle?.setAttribute("aria-expanded", "false");
  });
});

const revealElements = document.querySelectorAll(".reveal");
if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { rootMargin: "0px 0px -8%", threshold: 0.08 },
  );
  revealElements.forEach((element) => observer.observe(element));
} else {
  revealElements.forEach((element) => element.classList.add("visible"));
}

const leaderboardData = [
  { rank: 1, model: "Kimi-K3", group: "gt100", c: 52.38, p: 45.25, f: 2.38, k: 53.65 },
  { rank: 2, model: "Gemini-3.1-Pro", group: "proprietary", c: 50.37, p: 47.44, f: 2.19, k: 51.5 },
  { rank: 3, model: "GPT-5.5", group: "proprietary", c: 50.18, p: 47.17, f: 2.65, k: 51.55 },
  { rank: 4, model: "Gemini-3.5-Flash", group: "proprietary", c: 48.63, p: 49.36, f: 2.01, k: 49.63 },
  { rank: 5, model: "GPT-5.6-Sol", group: "proprietary", c: 44.97, p: 52.19, f: 2.83, k: 46.28 },
  { rank: 6, model: "Claude-Opus-4.8", group: "proprietary", c: 44.15, p: 50.46, f: 5.39, k: 46.67 },
  { rank: 7, model: "Hy3", group: "gt100", c: 42.6, p: 54.02, f: 3.38, k: 44.09 },
  { rank: 8, model: "Nemotron-3-Ultra", group: "gt100", c: 38.12, p: 57.95, f: 3.93, k: 39.68 },
  { rank: 9, model: "Claude-Sonnet-5", group: "proprietary", c: 37.84, p: 56.12, f: 6.03, k: 40.27 },
  { rank: 10, model: "GLM-5.2", group: "gt100", c: 37.29, p: 57.59, f: 5.12, k: 39.31 },
  { rank: 11, model: "MiniMax-M3", group: "gt100", c: 36.2, p: 55.94, f: 7.86, k: 39.29 },
  { rank: 12, model: "Qwen3.5-397B", group: "gt100", c: 32.27, p: 59.23, f: 8.5, k: 35.26 },
  { rank: 13, model: "DeepSeek-V4-Pro", group: "gt100", c: 31.99, p: 64.08, f: 3.93, k: 33.3 },
  { rank: 14, model: "DeepSeek-V4-Flash", group: "gt100", c: 29.98, p: 64.17, f: 5.85, k: 31.84 },
  { rank: 15, model: "Qwen3.5-122B", group: "gt100", c: 28.24, p: 57.22, f: 14.53, k: 33.05 },
  { rank: 16, model: "GPT-OSS-120B", group: "gt100", c: 22.49, p: 53.02, f: 24.5, k: 29.78 },
  { rank: 17, model: "Qwen3.5-35B", group: "10to100", c: 21.02, p: 55.94, f: 23.03, k: 27.32 },
  { rank: 18, model: "Qwen3.5-27B", group: "10to100", c: 19.47, p: 55.48, f: 25.05, k: 25.98 },
  { rank: 19, model: "Qwen3.6-35B", group: "10to100", c: 17.0, p: 59.32, f: 23.67, k: 22.28 },
  { rank: 20, model: "Seed-OSS-36B", group: "10to100", c: 15.36, p: 63.44, f: 21.21, k: 19.49 },
  { rank: 21, model: "OLMo-3.1-32B", group: "10to100", c: 14.99, p: 60.69, f: 24.31, k: 19.81 },
  { rank: 22, model: "Gemma-4-26B", group: "10to100", c: 13.71, p: 53.84, f: 32.45, k: 20.3 },
  { rank: 23, model: "Qwen3.5-9B", group: "lt10", c: 12.25, p: 46.25, f: 41.5, k: 20.94 },
  { rank: 24, model: "Gemma-4-31B", group: "10to100", c: 11.88, p: 58.59, f: 29.52, k: 16.86 },
  { rank: 25, model: "Qwen3.6-27B", group: "10to100", c: 10.15, p: 61.79, f: 28.06, k: 14.1 },
  { rank: 26, model: "Qwen3.5-4B", group: "lt10", c: 9.69, p: 39.03, f: 51.28, k: 19.89 },
  { rank: 27, model: "GPT-OSS-20B", group: "10to100", c: 9.14, p: 49.27, f: 41.59, k: 15.65 },
  { rank: 28, model: "Gemma-4-12B", group: "10to100", c: 5.85, p: 42.5, f: 51.65, k: 12.1 },
  { rank: 29, model: "OLMo-3-7B", group: "lt10", c: 5.21, p: 38.03, f: 56.76, k: 12.05 },
  { rank: 30, model: "Gemma-4-E4B", group: "lt10", c: 2.74, p: 32.82, f: 64.44, k: 7.71 },
  { rank: 31, model: "Qwen3.5-2B", group: "lt10", c: 1.65, p: 17.0, f: 81.35, k: 8.82 },
  { rank: 32, model: "Gemma-4-E2B", group: "lt10", c: 1.37, p: 24.31, f: 74.31, k: 5.34 },
];

const groupLabels = {
  en: {
    proprietary: "Proprietary",
    gt100: "Open >100B",
    "10to100": "Open 10B–100B",
    lt10: "Open <10B",
  },
  zh: {
    proprietary: "闭源模型",
    gt100: "开源 >100B",
    "10to100": "开源 10B–100B",
    lt10: "开源 <10B",
  },
};

const zhTranslations = {
  "Skip to content": "跳至正文",
  Overview: "概览",
  Construction: "构建流程",
  Benchmark: "基准",
  Leaderboard: "排行榜",
  Findings: "主要发现",
  Resources: "资源",
  "Toggle navigation": "展开或收起导航",
  "A closed-book knowledge probe": "闭卷知识探针",
  "Blind Men and the Elephant: Probing the Epistemic Myopia of LLMs under Long-Tail Divergent Knowledge":
    "盲人摸象：探究大模型在长尾分歧知识下的认知短视",
  "Can a language model remember a long-tail fact—and recover": "大模型能否记住一个长尾事实，并完整恢复",
  "all of its different verified accounts": "这一事实的不同已验证口径",
  "Tsinghua University ·": "清华大学 ·",
  "Tencent Youtu Lab ·": "腾讯优图实验室 ·",
  "University of Warwick": "华威大学",
  "* Equal contribution † Corresponding authors": "* 共同一作　† 通讯作者",
  Code: "代码",
  Dataset: "数据集",
  "Source corpus": "源语料",
  "Benchmark at a glance": "基准概览",
  "Open full resolution ↗": "查看高清图 ↗",
  "conflict subgraphs": "冲突子图",
  questions: "问题",
  "knowledge fields": "知识领域",
  "model configurations": "模型配置",
  "The problem": "问题",
  "Knowing one answer is not the same as knowing the whole fact.": "记住一个答案，不等于完整记住这个事实。",
  "Factual QA usually assumes one canonical answer. Long-tail web knowledge is different: evidence is sparse, sources may preserve incompatible accounts, and a model can recall the prevailing view while silently omitting another verified account.":
    "事实问答通常假设只有一个标准答案。长尾网络知识却并非如此：相关证据稀疏，不同来源可能保留互不相容的口径，而模型可能只记住主流说法，却遗漏另一种经过验证的口径。",
  "Long-tail QA": "长尾问答",
  "Was the rare fact recalled?": "模型记住了这个长尾事实吗？",
  "Useful for retention, but usually scores against a single canonical answer.": "可用于评测知识记忆，但通常只按一个标准答案评分。",
  "Do different accounts coexist in memory?": "不同口径是否共同保存在模型记忆中？",
  "Sources are withheld. The model must recover every verified account from parametric memory.":
    "评测时不提供来源，模型必须仅凭参数化记忆恢复所有已验证口径。",
  "Open-book conflict benchmarks": "开卷知识冲突基准",
  "Can supplied evidence be reconciled?": "模型能否整合给定的冲突证据？",
  "Useful for reading and reasoning, but typically exposes the conflict at inference time.":
    "适合评测阅读与推理能力，但通常会在推理时直接提供冲突证据。",
  "Epistemic myopia": "认知短视",
  "A response may be locally correct yet globally incomplete: the model remembers one documented account and presents it as the whole story.":
    "一个回答可能局部正确、整体却不完整：模型只记住一种有文献支持的口径，却把它当作事实全貌。",
  "Benchmark construction": "基准构建",
  "From a low-exposure corpus to auditable knowledge probes.": "从低暴露语料构建可审计的知识探针。",
  "A graph-based pipeline discovers naturally occurring disagreements, gathers support on both sides, and converts each verified conflict into a matched question pair.":
    "图构建流程发现自然存在的分歧，收集冲突两侧的支持证据，并将每个验证后的冲突转化为一对匹配问题。",
  Discover: "发现",
  "Start from": "始于",
  "Mine the filtered remainder of a web corpus, where low-exposure facts are less saturated by repetition.":
    "从网络语料过滤后的剩余部分挖掘数据，其中低暴露事实较少受到重复增强。",
  Link: "关联",
  "Retrieve related documents": "召回相关文档",
  "Knowledge-point tags form precise local clusters; normalized entities recover related documents across clusters.":
    "知识点标签形成精确的局部簇，归一化实体则跨簇召回相关文档。",
  Classify: "分类",
  "Build the document graph": "构建文档图",
  "An LLM reads full document pairs and induces support and conflict edges for the same subject–attribute pair.":
    "大模型阅读文档全文对，为同一主体—属性对判定支持边与冲突边。",
  Synthesize: "合成",
  "Sample subgraphs and generate QA": "采样子图并生成问答",
  "Each sampled conflict edge is expanded with support neighbors. Its local subgraph yields one named-entity and one clue-based question.":
    "每条采样的冲突边都会加入其支持邻居，并由局部子图生成一道实体题和一道线索题。",
  Verify: "验证",
  "Validate every account": "验证每一种口径",
  "An independent LLM checks the full seed documents, a web agent seeks external evidence, and human reviewers conduct the final audit.":
    "独立大模型检查种子文档全文，网络智能体搜索外部证据，最后由人工完成审核。",
  Probe: "探测",
  "Ask without showing the sources": "隐藏来源后进行提问",
  "Each conflict yields a named-entity question and a clue-based question with the same verified answer set.":
    "每个冲突生成一道实体题和一道线索题，两种表述共享同一组已验证答案。",
  "Discovery graph": "发现阶段文档图",
  "Hidden at evaluation time": "评测时隐藏",
  "Question only": "仅提供问题",
  "What one item looks like": "单条样例",
  "Two sources. Two accounts. One closed-book test.": "两个来源，两种口径，一次闭卷测试。",
  "Source documents establish the reference answers during construction, but are concealed from every evaluated model.":
    "来源文档在构建阶段用于确定参考答案，但评测时对所有模型隐藏。",
  "Source A": "来源 A",
  "Source B": "来源 B",
  "“Mother Teresa was born on": "“特蕾莎修女出生于",
  "… as Agnes Gonxha Bojaxhiu.”": "……原名阿格尼丝·贡扎·博亚久。”",
  "“Mother Teresa, Agnes Gonxha Bojaxhiu, was born in Skopje, Macedonia on":
    "“特蕾莎修女，原名阿格尼丝·贡扎·博亚久，于",
  ".”": "出生在马其顿斯科普里。”",
  "August 26, 1910": "1910 年 8 月 26 日",
  "August 27, 1910": "1910 年 8 月 27 日",
  "Named-entity formulation": "实体表述",
  "What birth date was reported for Mother Teresa?": "资料中报道的特蕾莎修女出生日期是什么？",
  "Clue-based formulation": "线索表述",
  "What birth date was reported for the Albanian-born Roman Catholic nun who founded the Missionaries of Charity and received the 1979 Nobel Peace Prize?":
    "这位出生于阿尔巴尼亚、创立仁爱传教修女会并获得 1979 年诺贝尔和平奖的罗马天主教修女，据报道出生于哪一天？",
  "Verified answer set": "已验证答案集",
  and: "和",
  "sources withheld": "来源已隐藏",
  "Diagnostic scoring": "诊断式评分",
  "Separate accessibility from completeness.": "区分知识可及性与记忆完整性。",
  "Instead of collapsing behavior into one accuracy number, ElephantBench partitions every response into complete, partial, or failed recall, with":
    "ElephantBench 不把模型表现压缩成单一准确率，而是将每个回答划分为完整召回、部分召回或召回失败，并满足",
  "Complete recall": "完整召回",
  "Every verified account is recovered without a material contradiction.": "恢复全部已验证口径，且没有实质性矛盾。",
  "Higher is better": "越高越好",
  "Partial recall": "部分召回",
  "At least one, but not every, verified account is recovered.": "恢复了至少一种、但并非全部已验证口径。",
  "Lower is better": "越低越好",
  "Failed recall": "召回失败",
  "No verified account is recovered, or the response materially contradicts the references.": "未恢复任何已验证口径，或回答与参考内容存在实质性矛盾。",
  "Conditional completeness": "条件完整度",
  "measures the share of complete recall among questions where at least one verified account is remembered.":
    "衡量在至少记住一种已验证口径的题目中，实现完整召回的比例。",
  "Model leaderboard": "模型排行榜",
  "Complete recall remains below 54% for every single model.": "所有单模型的完整召回率仍低于 54%。",
  "Main configurations reported in the paper, evaluated on all 1,094 questions. Higher C and K and lower P and F indicate better performance.":
    "论文报告的主配置，评测覆盖全部 1,094 道题。C 和 K 越高、P 和 F 越低越好。",
  "All models": "全部模型",
  Proprietary: "闭源模型",
  "Open >100B": "开源 >100B",
  "Open 10B–100B": "开源 10B–100B",
  "Open <10B": "开源 <10B",
  "Sorted by complete recall": "按完整召回率排序",
  Rank: "排名",
  Model: "模型",
  Type: "类型",
  "Values are percentages. C/P/F denote complete, partial, and failed recall; K = C / (C + P) is conditional completeness.":
    "数值均为百分比。C/P/F 分别表示完整、部分和失败召回；K = C / (C + P) 表示条件完整度。",
  "Main findings": "主要发现",
  "Stronger recall still leaves much of the elephant unseen.": "即使记忆更强，模型看到的仍只是“大象”的一部分。",
  "Across 32 open-weight and proprietary model configurations, incomplete recall remains the dominant failure mode once a long-tail fact becomes accessible.":
    "在 32 个开源与闭源模型配置中，一旦长尾事实能够被记起，不完整召回仍是主要失败模式。",
  "Strongest single model": "最强单模型",
  complete: "完整召回",
  Complete: "完整",
  Partial: "部分",
  Failed: "失败",
  "Nearly all remaining questions are answered with only part of the verified account set.":
    "其余问题几乎都只恢复了已验证答案集的一部分。",
  "Models are complementary, but share a blind spot.": "模型之间具有互补性，却也共享盲点。",
  "A greedy oracle over all configurations raises complete recall substantially, yet 18.8% of questions remain partial for every model.":
    "对全部配置进行贪心组合可显著提升完整召回率，但仍有 18.8% 的问题对所有模型而言都只能部分召回。",
  Scale: "模型规模",
  "Scaling improves recall, but incompleteness persists.": "规模扩展提升召回，但不完整性依然存在。",
  "Larger models generally reduce complete failure, yet many questions still recover only part of the verified account set.":
    "更大的模型通常能减少召回失败，但许多问题仍只恢复了已验证答案集的一部分。",
  Reasoning: "推理",
  "More deliberation is not a uniform remedy.": "增加推理并非普遍有效。",
  "Reasoning improves complete recall for some frontier models, but can suppress the less salient account in smaller models.":
    "推理可提升部分前沿模型的完整召回，却可能让较小模型压制不够显著的口径。",
  "Exposure asymmetry": "暴露不对称",
  "The rare account determines whether memory is complete.": "稀有口径决定记忆是否完整。",
  "Exposure to the prevailing view helps a model remember the fact. Exposure to its less frequently reported counterpart is more closely associated with recalling the full account set.":
    "更多接触主流口径有助于模型记住该事实，而对较少被报道口径的暴露与完整恢复全部口径的关系更紧密。",
  "Prevailing view": "主流口径",
  "fact accessibility": "事实可及性",
  "Rare account": "稀有口径",
  "memory completeness": "记忆完整性",
  "Use ElephantBench": "使用 ElephantBench",
  "Data, evaluation, and construction in one repository.": "数据、评测与构建代码汇集于同一仓库。",
  "Run target models closed-book, grade responses with the published rubric, compute C/P/F/K, or reconstruct the benchmark from the source corpus.":
    "以闭卷方式运行待测模型，使用公开评分标准评判回答并计算 C/P/F/K，或从源语料重新构建基准。",
  "Evaluation code": "评测代码",
  "Benchmark data": "基准数据",
  "Quick start": "快速开始",
  Citation: "引用",
  "Build on the probe.": "基于这一知识探针继续研究。",
  "Copy BibTeX": "复制 BibTeX",
  Copied: "已复制",
  "Select and copy": "请选择并复制",
  "A closed-book probe of long-tail factual memory and cross-source completeness.":
    "用于评测长尾事实记忆及跨来源完整性的闭卷知识探针。",
};

const normalizeText = (value) => value.trim().replace(/\s+/g, " ");
const translatableTextNodes = [];
const textWalker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
let currentTextNode = textWalker.nextNode();
while (currentTextNode) {
  const parent = currentTextNode.parentElement;
  const key = normalizeText(currentTextNode.nodeValue ?? "");
  if (key && zhTranslations[key] && !parent?.closest("pre, code, script, style")) {
    translatableTextNodes.push({ node: currentTextNode, original: currentTextNode.nodeValue, key });
  }
  currentTextNode = textWalker.nextNode();
}

const languageButtons = document.querySelectorAll("[data-language]");
const leaderboardBody = document.querySelector("[data-leaderboard-body]");
const leaderboardFilters = document.querySelectorAll("[data-leaderboard-filter]");
const copyButton = document.querySelector("[data-copy-citation]");
let activeLanguage = "en";
let activeLeaderboardFilter = "all";

const renderLeaderboard = () => {
  if (!leaderboardBody) return;
  const rows = leaderboardData.filter(
    (entry) => activeLeaderboardFilter === "all" || entry.group === activeLeaderboardFilter,
  );
  leaderboardBody.innerHTML = rows
    .map(
      (entry) => `
        <tr class="${entry.rank <= 3 ? "leaderboard-top" : ""}">
          <td class="leaderboard-rank"><span>${entry.rank}</span></td>
          <td class="leaderboard-model">${entry.model}</td>
          <td class="leaderboard-type"><span>${groupLabels[activeLanguage][entry.group]}</span></td>
          <td class="leaderboard-value leaderboard-value-primary">${entry.c.toFixed(2)}</td>
          <td class="leaderboard-value">${entry.p.toFixed(2)}</td>
          <td class="leaderboard-value">${entry.f.toFixed(2)}</td>
          <td class="leaderboard-value leaderboard-value-primary">${entry.k.toFixed(2)}</td>
        </tr>`,
    )
    .join("");
};

const localizedAttributes = {
  en: {
    title: "ElephantBench — A Closed-Book Knowledge Probe",
    description:
      "ElephantBench is a closed-book knowledge probe for evaluating whether LLMs remember long-tail facts and their different verified accounts.",
    language: "Language",
    navigation: "Primary navigation",
    projectLinks: "Project links",
    statistics: "Benchmark statistics",
    filters: "Filter models",
    figureAlt: "ElephantBench construction, conflict example, and closed-book evaluation overview",
  },
  zh: {
    title: "ElephantBench — 闭卷知识探针",
    description: "ElephantBench 是一个闭卷知识探针，用于评测大模型能否记住长尾事实及其不同的已验证口径。",
    language: "语言",
    navigation: "主导航",
    projectLinks: "项目链接",
    statistics: "基准统计",
    filters: "筛选模型",
    figureAlt: "ElephantBench 构建流程、冲突样例与闭卷评测概览",
  },
};

const applyLanguage = (language) => {
  activeLanguage = language === "zh" ? "zh" : "en";
  document.documentElement.lang = activeLanguage === "zh" ? "zh-CN" : "en";

  translatableTextNodes.forEach(({ node, original, key }) => {
    if (activeLanguage === "en") {
      node.nodeValue = original;
      return;
    }
    const leading = original.match(/^\s*/)?.[0] ?? "";
    const trailing = original.match(/\s*$/)?.[0] ?? "";
    node.nodeValue = `${leading}${zhTranslations[key]}${trailing}`;
  });

  languageButtons.forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.language === activeLanguage));
  });

  const labels = localizedAttributes[activeLanguage];
  document.title = labels.title;
  document.querySelector('meta[name="description"]')?.setAttribute("content", labels.description);
  document.querySelector(".language-switcher")?.setAttribute("aria-label", labels.language);
  nav?.setAttribute("aria-label", labels.navigation);
  document.querySelector(".hero-actions")?.setAttribute("aria-label", labels.projectLinks);
  document.querySelector(".hero-stats")?.setAttribute("aria-label", labels.statistics);
  document.querySelector(".leaderboard-filters")?.setAttribute("aria-label", labels.filters);
  document.querySelector(".hero-visual img")?.setAttribute("alt", labels.figureAlt);

  if (copyButton) {
    copyButton.textContent = activeLanguage === "zh" ? zhTranslations["Copy BibTeX"] : "Copy BibTeX";
  }

  renderLeaderboard();
  try {
    window.localStorage.setItem("elephantbench-language", activeLanguage);
  } catch {
    // The page still works when storage is unavailable.
  }
};

leaderboardFilters.forEach((button) => {
  button.addEventListener("click", () => {
    activeLeaderboardFilter = button.dataset.leaderboardFilter ?? "all";
    leaderboardFilters.forEach((candidate) => {
      const isActive = candidate === button;
      candidate.classList.toggle("active", isActive);
      candidate.setAttribute("aria-pressed", String(isActive));
    });
    renderLeaderboard();
  });
});

languageButtons.forEach((button) => {
  button.addEventListener("click", () => applyLanguage(button.dataset.language));
});

let initialLanguage = "en";
try {
  initialLanguage = window.localStorage.getItem("elephantbench-language") === "zh" ? "zh" : "en";
} catch {
  initialLanguage = "en";
}
applyLanguage(initialLanguage);

copyButton?.addEventListener("click", async () => {
  const citation = document.querySelector("#bibtex")?.textContent?.trim();
  if (!citation) return;

  try {
    await navigator.clipboard.writeText(citation);
    copyButton.textContent = activeLanguage === "zh" ? zhTranslations.Copied : "Copied";
  } catch {
    copyButton.textContent = activeLanguage === "zh" ? zhTranslations["Select and copy"] : "Select and copy";
  }

  window.setTimeout(() => {
    copyButton.textContent = activeLanguage === "zh" ? zhTranslations["Copy BibTeX"] : "Copy BibTeX";
  }, 1800);
});
