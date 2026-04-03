# Streamlit UI Specifications for RAG System

## 📋 Overview
Modern, user-friendly web interface for the RAG System using Streamlit.

---

## 🎨 UI Layout

### **Sidebar (Left Panel)**
```
┌─────────────────────┐
│  RAG MCP PROJECT    │
│  ═══════════════    │
│                     │
│  📊 Dashboard       │
│  💬 Chat Interface  │
│  📁 Documents       │
│  🗂️ Cache Manager   │
│  ⚙️ Settings        │
│                     │
│  ───────────────    │
│  System Status:     │
│  ● Active           │
│  📈 Queries: 45     │
│  💾 Cache: 85%      │
└─────────────────────┘
```

---

## 🏠 Page 1: Dashboard

### **Metrics Cards (Top Row)**
```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ 📊 Total     │  │ 💾 Cache     │  │ ⚡ Avg       │  │ 📁 Documents │
│    Queries   │  │    Hit Rate  │  │    Response  │  │    Indexed   │
│              │  │              │  │    Time      │  │              │
│    245       │  │    87.5%     │  │    2.3s      │  │    5         │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

### **Charts Section**
- **Line Chart:** Queries over time (last 24h)
- **Bar Chart:** Most queried topics
- **Pie Chart:** Cache hits vs misses

### **Recent Queries Table**
| Time | Question | Cached | Response Time | Sources |
|------|----------|--------|---------------|---------|
| 10:23 | What is 4M? | ✅ | 0.001s | 1 |
| 10:20 | Lean concepts | ❌ | 15.2s | 3 |

---

## 💬 Page 2: Chat Interface

### **Main Layout**
```
┌─────────────────────────────────────────────────────────────┐
│  💬 Chat with your Documents                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🤖 RAG Assistant                                           │
│  Bonjour! Ask me anything about your documents.            │
│                                                             │
│  ───────────────────────────────────────────────────────── │
│                                                             │
│  👤 You                                                     │
│  What is the stability of 4M?                               │
│                                                             │
│  ───────────────────────────────────────────────────────── │
│                                                             │
│  🤖 RAG Assistant                           ⚡ Cached: Yes  │
│  According to the context, the stability of 4M is...       │
│                                                             │
│  📚 Sources:                                                │
│  • Chapitre 5 ODQ 2025.pdf (chunk 2)                       │
│  • Chapitre 2 ODQ 2025.pdf (chunk 6)                       │
│                                                             │
│  ⏱️ Response time: 0.002s                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
│  Type your question here...                    [Send 📤]    │
└─────────────────────────────────────────────────────────────┘
```

### **Features:**
- ✅ Chat history (session-based, clearable)
- ✅ Copy answer button
- ✅ Export conversation (JSON/TXT)
- ✅ Source citations clickable (show chunk preview)
- ✅ Loading spinner during query
- ✅ Clear conversation button
- ✅ Auto-scroll to latest message

---

## 📁 Page 3: Documents Manager

### **Upload Section**
```
┌─────────────────────────────────────────────────────────────┐
│  📤 Upload New Documents                                     │
├─────────────────────────────────────────────────────────────┤
│  Drag & Drop or Click to Upload                             │
│  Supported: .txt, .pdf, .docx, .md                          │
│  [Browse Files]                                             │
└─────────────────────────────────────────────────────────────┘
```

### **Documents List**
```
┌─────────────────────────────────────────────────────────────┐
│  📚 Indexed Documents (5 total)                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📄 Chapitre 1 ODQ 2025.pdf               125 KB  🗑️ 👁️   │
│     Indexed: Feb 18, 2026 09:15                             │
│     Chunks: 45                                              │
│                                                             │
│  ───────────────────────────────────────────────────────── │
│                                                             │
│  📄 Chapitre 2 ODQ 2025.pdf               98 KB   🗑️ 👁️   │
│     Indexed: Feb 18, 2026 09:15                             │
│     Chunks: 38                                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘

[🔄 Reindex All Documents]
```

### **Features:**
- ✅ Upload new documents
- ✅ View document details
- ✅ Delete documents (with confirmation)
- ✅ Reindex all button
- ✅ Document preview modal
- ✅ Show indexing progress

---

## 🗂️ Page 4: Cache Manager

### **Cache Statistics**
```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ 📊 Total     │  │ ✅ Cache     │  │ ❌ Cache     │
│    Queries   │  │    Hits      │  │    Misses    │
│    150       │  │    125       │  │    25        │
└──────────────┘  └──────────────┘  └──────────────┘

┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ 💾 Hit Rate  │  │ 📦 Entries   │  │ 🕐 TTL       │
│    83.3%     │  │    85        │  │    60 min    │
└──────────────┘  └──────────────┘  └──────────────┘
```

### **Cache Entries Table**
| Question | Hits | Last Used | Size | Actions |
|----------|------|-----------|------|---------|
| What is 4M? | 5 | 2 min ago | 2.3 KB | 🗑️ |
| Lean Manufacturing | 3 | 5 min ago | 3.1 KB | 🗑️ |

### **Actions:**
```
[🗑️ Clear All Cache]  [📊 Export Cache Stats]  [🔄 Refresh]
```

---

## ⚙️ Page 5: Settings

### **LLM Configuration**
```
┌─────────────────────────────────────────────────────────────┐
│  🤖 LLM Settings                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Model:  ○ Google Gemini  ● Local Ollama                   │
│                                                             │
│  Temperature: [====|----------] 0.1                         │
│                                                             │
│  Max Tokens: 2048                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### **Cache Settings**
```
┌─────────────────────────────────────────────────────────────┐
│  💾 Cache Configuration                                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TTL (minutes): [60  ]                                      │
│                                                             │
│  Max Entries:   [1000]                                      │
│                                                             │
│  Persistence:   ☑ Enabled                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### **System Info**
```
┌─────────────────────────────────────────────────────────────┐
│  ℹ️ System Information                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Version:        1.0.0                                      │
│  Python:         3.12.0                                     │
│  API Status:     🟢 Running (http://localhost:8000)         │
│  Vector Store:   ChromaDB                                   │
│  Embeddings:     sentence-transformers                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎨 Design Specifications

### **Color Scheme**
```python
PRIMARY_COLOR = "#1E88E5"      # Blue
SECONDARY_COLOR = "#43A047"    # Green
ACCENT_COLOR = "#FDD835"       # Yellow
ERROR_COLOR = "#E53935"        # Red
SUCCESS_COLOR = "#43A047"      # Green
BACKGROUND = "#FAFAFA"         # Light gray
TEXT_PRIMARY = "#212121"       # Dark gray
TEXT_SECONDARY = "#757575"     # Medium gray
```

### **Typography**
- **Headings:** Sans-serif, bold
- **Body:** Sans-serif, regular
- **Code/Data:** Monospace

### **Icons**
- 📊 Dashboard
- 💬 Chat
- 📁 Documents
- 🗂️ Cache
- ⚙️ Settings
- ✅ Success
- ❌ Error
- ⚡ Fast/Cached
- 🤖 AI/Assistant
- 👤 User

---

## 🔧 Technical Requirements

### **Dependencies**
```python
streamlit >= 1.51.0
plotly >= 5.14.0  # For interactive charts
pandas >= 2.0.0   # For data tables
requests >= 2.31.0  # For API calls
```

### **API Integration**
- **Base URL:** `http://localhost:8000`
- **Endpoints:**
  - `GET /health` - System status
  - `POST /query` - Send questions
  - `GET /cache/stats` - Cache statistics
  - `DELETE /cache/clear` - Clear cache
  - `GET /documents` - List documents
  - `POST /documents/reindex` - Reindex

### **State Management**
```python
st.session_state = {
    'chat_history': [],       # List of {role, content, sources, time}
    'current_page': 'chat',   # Active page
    'settings': {},           # User preferences
    'api_url': 'http://localhost:8000'
}
```

---

## 📱 Responsive Design
- Mobile-friendly (tabs instead of sidebar on mobile)
- Adaptive layout for different screen sizes
- Touch-friendly buttons

---

## 🚀 Performance
- Lazy loading for charts
- Pagination for large tables
- Debounced search inputs
- Async API calls (non-blocking UI)

---

## 🔒 Security
- No API keys in frontend code
- API calls through localhost only
- Input sanitization
- CORS handling

---

## ✨ Nice-to-Have Features
- 🌙 Dark mode toggle
- 📊 Export dashboard as PDF
- 🔔 Notifications for long queries
- 📝 Query suggestions based on history
- 🎯 Advanced filters (date range, document source)
- 🔍 Full-text search in documents
- 📈 Real-time metrics updates
- 🎨 Customizable themes

---

## 📝 Notes
- Keep UI simple and intuitive
- Focus on chat interface as main feature
- Provide clear feedback for all actions
- Handle errors gracefully with user-friendly messages
- Maintain consistency throughout the UI
