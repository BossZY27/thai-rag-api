const healthStatus = document.querySelector("#healthStatus");
const apiKeyInput = document.querySelector("#apiKey");
const documentsEl = document.querySelector("#documents");
const documentCountEl = document.querySelector("#documentCount");
const categoriesEl = document.querySelector("#categories");
const categoryCountEl = document.querySelector("#categoryCount");
const selectedCategoryLabel = document.querySelector("#selectedCategoryLabel");
const selectedCategoryLevel = document.querySelector("#selectedCategoryLevel");
const messagesEl = document.querySelector("#messages");
const chatForm = document.querySelector("#chatForm");
const questionInput = document.querySelector("#question");
const modeButtons = document.querySelectorAll(".modeButton");
const batchQuestionsInput = document.querySelector("#batchQuestions");
const runBatchButton = document.querySelector("#runBatchButton");
const loadLogsButton = document.querySelector("#loadLogsButton");
const labResultsEl = document.querySelector("#labResults");

let selectedCategory = "";
let categoryLabels = {};
let answerMode = "fast";
let lastExchange = null;

function headers(extra = {}) {
  const apiKey = apiKeyInput.value.trim();
  return {
    ...(apiKey ? { "x-api-key": apiKey } : {}),
    ...extra,
  };
}

function setStatus(text, type = "") {
  healthStatus.textContent = text;
  healthStatus.className = `status ${type}`.trim();
}

function levelText(level) {
  if (level === 1) return "ระดับ 1: หมวดสำคัญ";
  if (level === 2) return "ระดับ 2: หมวดย่อย";
  return "ระดับ 3: ข้อมูลประกอบ";
}

function addMessage(role, text, sources = [], meta = {}) {
  const message = document.createElement("div");
  message.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  if (sources.length) {
    const sourceEl = document.createElement("div");
    sourceEl.className = "sources";
    const unique = sources
      .map((source) => {
        const category = categoryLabels[source.category] || source.subcategory || source.category || "ไม่ระบุหมวด";
        return `${source.title} หน้า ${source.page} · ${category}`;
      })
      .filter((value, index, list) => list.indexOf(value) === index);
    sourceEl.textContent = `อ้างอิง: ${unique.join(", ")}`;
    bubble.appendChild(sourceEl);
  }

  if (role === "assistant" && meta.debug) {
    bubble.appendChild(renderDebug(meta));
  }

  if (role === "assistant" && meta.feedback) {
    bubble.appendChild(renderFeedback(meta.feedback));
  }

  message.appendChild(bubble);
  messagesEl.appendChild(message);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return message;
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error("Health check failed");
    setStatus(`พร้อมใช้: ${data.chat_model} + ${data.embed_model}`, "ok");
  } catch (error) {
    setStatus("API ยังไม่พร้อม", "bad");
  }
}

function updateSelectedCategory(category) {
  selectedCategory = category;
  document.querySelectorAll(".categoryCard").forEach((card) => {
    card.classList.toggle("active", card.dataset.category === selectedCategory);
  });

  if (!selectedCategory) {
    selectedCategoryLabel.textContent = "ทุกหมวดข้อมูล";
    selectedCategoryLevel.textContent = "ค้นทุกระดับ";
    return;
  }

  const active = document.querySelector(`.categoryCard[data-category="${selectedCategory}"]`);
  selectedCategoryLabel.textContent = categoryLabels[selectedCategory] || selectedCategory;
  selectedCategoryLevel.textContent = active?.dataset.levelText || "เลือกหมวดแล้ว";
}

async function loadCategories() {
  try {
    const response = await fetch("/categories", { headers: headers() });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "โหลดหมวดไม่สำเร็จ");

    const categories = data.categories || [];
    categoryLabels = Object.fromEntries(categories.map((category) => [category.id, category.label]));
    categoryCountEl.textContent = `${categories.filter((category) => category.chunks > 0).length} หมวด`;
    categoriesEl.innerHTML = "";

    const allCard = document.createElement("button");
    allCard.type = "button";
    allCard.className = "categoryCard active";
    allCard.dataset.category = "";
    allCard.innerHTML = `
      <span class="categoryName">ทุกหมวดข้อมูล</span>
      <span class="categoryMeta">ให้ AI เลือกเองจากทั้งฐาน</span>
    `;
    allCard.addEventListener("click", () => updateSelectedCategory(""));
    categoriesEl.appendChild(allCard);

    for (const category of categories) {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "categoryCard";
      card.dataset.category = category.id;
      card.dataset.levelText = levelText(category.level);
      card.innerHTML = `
        <span class="categoryName"></span>
        <span class="categoryMeta">${levelText(category.level)} · ${category.chunks} ชิ้นข้อมูล</span>
      `;
      card.querySelector(".categoryName").textContent = category.label;
      card.addEventListener("click", () => updateSelectedCategory(category.id));
      categoriesEl.appendChild(card);
    }
  } catch (error) {
    categoriesEl.textContent = error.message;
  }
}

async function loadDocuments() {
  try {
    const response = await fetch("/documents", { headers: headers() });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "โหลดฐานความรู้ไม่สำเร็จ");

    const documents = data.documents || [];
    documentCountEl.textContent = `${documents.length} ชุด`;
    if (!documents.length) {
      documentsEl.textContent = "ยังไม่มีข้อมูล";
      return;
    }

    documentsEl.innerHTML = "";
    for (const doc of documents) {
      const item = document.createElement("div");
      item.className = "docItem";
      item.innerHTML = `
        <div class="docName"></div>
        <div class="docMeta">${doc.source_type} · ${doc.pages_with_text} หน้า · ${doc.chunks} ชิ้นข้อมูล</div>
        <div class="docCategory"></div>
      `;
      item.querySelector(".docName").textContent = doc.title;
      item.querySelector(".docCategory").textContent = categoryLabels[doc.category] || doc.category || "ไม่ระบุหมวด";
      documentsEl.appendChild(item);
    }
  } catch (error) {
    documentsEl.textContent = error.message;
  }
}

function scoreText(value) {
  if (value === undefined || value === null) return "-";
  return Number(value).toFixed(3);
}

function uniqueSourceLabels(sources = []) {
  return sources
    .map((source) => {
      const category = categoryLabels[source.category] || source.subcategory || source.category || "ไม่ระบุหมวด";
      return `${source.title} หน้า ${source.page} · ${category}`;
    })
    .filter((value, index, list) => list.indexOf(value) === index);
}

function isSourceFollowUp(question) {
  const compact = question.replace(/\s+/g, "").toLowerCase();
  const sourceWords = ["หน้าไหน", "จากหน้าไหน", "มาจากไหน", "ข้อมูลจากไหน", "เอามาจากไหน", "เอาข้อมูลมาจากไหน", "ไปเอาข้อมูล", "อ้างอิง", "แหล่งข้อมูล", "ที่มา"];
  const followUpWords = ["ข้อมูลนี้", "คำตอบนี้", "อันนี้", "ตรงนี้", "เมื่อกี้", "ที่ตอบ", "คุณ", "ai", "ระบบ"];
  return sourceWords.some((word) => compact.includes(word)) && followUpWords.some((word) => compact.includes(word));
}

function answerFromPreviousSources() {
  if (!lastExchange) {
    return "ยังไม่มีคำตอบก่อนหน้าให้ย้อนดูแหล่งอ้างอิงครับ ลองถามข้อมูลจากฐานความรู้ก่อนหนึ่งคำถาม";
  }

  const labels = uniqueSourceLabels(lastExchange.sources || []);
  if (!labels.length) {
    return "คำตอบก่อนหน้าไม่มีแหล่งอ้างอิง เพราะระบบประเมินว่ายังไม่พบข้อมูลที่ตอบคำถามนั้นโดยตรงในฐานข้อมูล";
  }

  return `คำตอบก่อนหน้าอ้างอิงจาก:\n${labels.map((label) => `- ${label}`).join("\n")}`;
}

function renderDebug(meta) {
  const debug = meta.debug || {};
  const details = document.createElement("details");
  details.className = "debugBox";

  const summary = document.createElement("summary");
  summary.textContent = `Debug: intent=${meta.intent || debug.intent || "-"} · log #${meta.logId || "-"}`;
  details.appendChild(summary);

  const terms = document.createElement("div");
  terms.className = "debugLine";
  terms.textContent = `คำค้นที่ขยาย: ${(debug.expanded_terms || []).join(", ") || "-"}`;
  details.appendChild(terms);

  const chunks = document.createElement("div");
  chunks.className = "debugChunks";
  for (const chunk of debug.top_chunks || []) {
    const item = document.createElement("div");
    item.className = "debugChunk";
    item.textContent = `${chunk.title} หน้า ${chunk.page} · ${categoryLabels[chunk.category] || chunk.subcategory || chunk.category} · score ${scoreText(chunk.score)} / semantic ${scoreText(chunk.semantic_score)} / keyword ${scoreText(chunk.keyword_score)}`;
    chunks.appendChild(item);
  }
  if (!chunks.children.length) {
    chunks.textContent = "ไม่มี chunk ที่ใช้ตอบ เพราะระบบตัดสินว่าไม่พบข้อมูลตรงคำถาม";
  }
  details.appendChild(chunks);
  return details;
}

function renderFeedback(payload) {
  const wrapper = document.createElement("div");
  wrapper.className = "feedbackBar";

  const options = [
    ["correct", "ถูก"],
    ["wrong", "ผิด"],
    ["should_unknown", "ควรตอบไม่พบ"],
  ];

  for (const [rating, label] of options) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.addEventListener("click", async () => {
      button.disabled = true;
      await submitFeedback({ ...payload, rating });
      button.textContent = "บันทึกแล้ว";
    });
    wrapper.appendChild(button);
  }

  return wrapper;
}

async function submitFeedback(payload) {
  await fetch("/feedback", {
    method: "POST",
    headers: headers({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      log_id: payload.logId,
      question: payload.question,
      answer: payload.answer,
      rating: payload.rating,
    }),
  });
}

function renderLabResults(title, items) {
  labResultsEl.innerHTML = "";
  const heading = document.createElement("div");
  heading.className = "labTitle";
  heading.textContent = title;
  labResultsEl.appendChild(heading);

  items.forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "labItem";
    row.innerHTML = `
      <div class="labQuestion"></div>
      <div class="labAnswer"></div>
      <div class="labMeta"></div>
    `;
    row.querySelector(".labQuestion").textContent = item.question || (item.id ? `#${item.id}` : `คำถามที่ ${index + 1}`);
    row.querySelector(".labAnswer").textContent = item.answer || "";
    row.querySelector(".labMeta").textContent = `intent: ${item.intent || "-"} · ${(item.elapsed_ms || 0) / 1000}s · sources: ${(item.sources || []).length}`;
    labResultsEl.appendChild(row);
  });
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage("user", question);
  questionInput.value = "";

  if (isSourceFollowUp(question)) {
    addMessage("assistant", answerFromPreviousSources());
    return;
  }

  addMessage("assistant", answerMode === "fast" ? "กำลังค้นคำตอบแบบเร็วจากฐานข้อมูล..." : "กำลังค้นข้อมูลและให้โมเดลเรียบเรียงคำตอบ...");

  const pending = messagesEl.lastElementChild;

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        question,
        top_k: answerMode === "fast" ? 10 : 8,
        category: selectedCategory || null,
        mode: answerMode,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "ถามไม่สำเร็จ");

    pending.remove();
    const elapsed = data.elapsed_ms ? `\n\nเวลาในการตอบ: ${(data.elapsed_ms / 1000).toFixed(1)} วินาที` : "";
    lastExchange = {
      logId: data.log_id,
      question,
      answer: data.answer || "ไม่มีคำตอบกลับมา",
      sources: data.sources || [],
    };
    addMessage(
      "assistant",
      `${data.answer || "ไม่มีคำตอบกลับมา"}${elapsed}`,
      data.sources || [],
      {
        logId: data.log_id,
        intent: data.intent,
        debug: data.debug,
        feedback: lastExchange,
      },
    );
  } catch (error) {
    pending.remove();
    addMessage("assistant", `ถามไม่สำเร็จ: ${error.message}`);
  }
});

apiKeyInput.addEventListener("change", async () => {
  await loadCategories();
  await loadDocuments();
});

modeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    answerMode = button.dataset.mode || "fast";
    modeButtons.forEach((item) => item.classList.toggle("active", item === button));
  });
});

runBatchButton.addEventListener("click", async () => {
  const questions = batchQuestionsInput.value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  if (!questions.length) {
    labResultsEl.textContent = "ใส่คำถามก่อน บรรทัดละ 1 คำถาม";
    return;
  }

  runBatchButton.disabled = true;
  labResultsEl.textContent = "กำลังรันทดสอบ...";
  try {
    const response = await fetch("/test/batch", {
      method: "POST",
      headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        questions,
        mode: answerMode,
        category: selectedCategory || null,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "รันทดสอบไม่สำเร็จ");
    renderLabResults(`ผลทดสอบ ${data.count} คำถาม`, data.results || []);
  } catch (error) {
    labResultsEl.textContent = `รันทดสอบไม่สำเร็จ: ${error.message}`;
  } finally {
    runBatchButton.disabled = false;
  }
});

loadLogsButton.addEventListener("click", async () => {
  labResultsEl.textContent = "กำลังโหลด log...";
  try {
    const response = await fetch("/logs?limit=20", { headers: headers() });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "โหลด log ไม่สำเร็จ");
    renderLabResults("Log ล่าสุด", data.logs || []);
  } catch (error) {
    labResultsEl.textContent = `โหลด log ไม่สำเร็จ: ${error.message}`;
  }
});

async function boot() {
  await checkHealth();
  await loadCategories();
  await loadDocuments();
}

boot();
